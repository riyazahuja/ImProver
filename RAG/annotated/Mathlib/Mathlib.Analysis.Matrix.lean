/-- Seminormed group instance (using sup norm of sup norm) for matrices over a seminormed group. Not
declared as an instance because there are several natural choices for defining the norm of a
matrix. -/
protected def seminormedAddCommGroup : SeminormedAddCommGroup (Matrix m n α) :=
  Pi.seminormedAddCommGroup



theorem norm_def (A : Matrix m n α) : ‖A‖ = ‖fun i j => A i j‖ := rfl


/-- The norm of a matrix is the sup of the sup of the nnnorm of the entries -/
lemma norm_eq_sup_sup_nnnorm (A : Matrix m n α) :
    ‖A‖ = Finset.sup Finset.univ fun i ↦ Finset.sup Finset.univ fun j ↦ ‖A i j‖₊ := by
  /-
    m : Type u_3
    n : Type u_4
    α : Type u_5
    inst✝² : Fintype m
    inst✝¹ : Fintype n
    inst✝ : SeminormedAddCommGroup α
    A : Matrix m n α
    ⊢ Eq (Norm.norm A) ↑(Finset.univ.sup fun i => Finset.univ.sup fun j => NNNorm. …
  -/
  simp_rw [Matrix.norm_def, Pi.norm_def, Pi.nnnorm_def]
  /-
    🎉 no goals
  -/


theorem nnnorm_def (A : Matrix m n α) : ‖A‖₊ = ‖fun i j => A i j‖₊ := rfl


theorem norm_le_iff {r : ℝ} (hr : 0 ≤ r) {A : Matrix m n α} : ‖A‖ ≤ r ↔ ∀ i j, ‖A i j‖ ≤ r := by
  /-
    m : Type u_3
    n : Type u_4
    α : Type u_5
    inst✝² : Fintype m
    inst✝¹ : Fintype n
    inst✝ : SeminormedAddCommGroup α
    r : Real
    hr : LE.le 0 r
    A : Matrix m n α
    ⊢ Iff (LE.le (Norm.norm A) r) (∀ (i : m) (j : n), LE.le (Norm.norm (A i j)) r)
  -/
  simp_rw [norm_def, pi_norm_le_iff_of_nonneg hr]
  /-
    🎉 no goals
  -/


theorem nnnorm_le_iff {r : ℝ≥0} {A : Matrix m n α} : ‖A‖₊ ≤ r ↔ ∀ i j, ‖A i j‖₊ ≤ r := by
  /-
    m : Type u_3
    n : Type u_4
    α : Type u_5
    inst✝² : Fintype m
    inst✝¹ : Fintype n
    inst✝ : SeminormedAddCommGroup α
    r : NNReal
    A : Matrix m n α
    ⊢ Iff (LE.le (NNNorm.nnnorm A) r) (∀ (i : m) (j : n), LE.le (NNNorm.nnnorm (A  …
  -/
  simp_rw [nnnorm_def, pi_nnnorm_le_iff]
  /-
    🎉 no goals
  -/


theorem norm_lt_iff {r : ℝ} (hr : 0 < r) {A : Matrix m n α} : ‖A‖ < r ↔ ∀ i j, ‖A i j‖ < r := by
  /-
    m : Type u_3
    n : Type u_4
    α : Type u_5
    inst✝² : Fintype m
    inst✝¹ : Fintype n
    inst✝ : SeminormedAddCommGroup α
    r : Real
    hr : LT.lt 0 r
    A : Matrix m n α
    ⊢ Iff (LT.lt (Norm.norm A) r) (∀ (i : m) (j : n), LT.lt (Norm.norm (A i j)) r)
  -/
  simp_rw [norm_def, pi_norm_lt_iff hr]
  /-
    🎉 no goals
  -/


theorem nnnorm_lt_iff {r : ℝ≥0} (hr : 0 < r) {A : Matrix m n α} :
    ‖A‖₊ < r ↔ ∀ i j, ‖A i j‖₊ < r := by
  /-
    m : Type u_3
    n : Type u_4
    α : Type u_5
    inst✝² : Fintype m
    inst✝¹ : Fintype n
    inst✝ : SeminormedAddCommGroup α
    r : NNReal
    hr : LT.lt 0 r
    A : Matrix m n α
    ⊢ Iff (LT.lt (NNNorm.nnnorm A) r) (∀ (i : m) (j : n), LT.lt (NNNorm.nnnorm (A  …
  -/
  simp_rw [nnnorm_def, pi_nnnorm_lt_iff hr]
  /-
    🎉 no goals
  -/


theorem norm_entry_le_entrywise_sup_norm (A : Matrix m n α) {i : m} {j : n} : ‖A i j‖ ≤ ‖A‖ :=
  (norm_le_pi_norm (A i) j).trans (norm_le_pi_norm A i)


theorem nnnorm_entry_le_entrywise_sup_nnnorm (A : Matrix m n α) {i : m} {j : n} : ‖A i j‖₊ ≤ ‖A‖₊ :=
  (nnnorm_le_pi_nnnorm (A i) j).trans (nnnorm_le_pi_nnnorm A i)


@[simp]
theorem nnnorm_map_eq (A : Matrix m n α) (f : α → β) (hf : ∀ a, ‖f a‖₊ = ‖a‖₊) :
    ‖A.map f‖₊ = ‖A‖₊ := by
  /-
    m : Type u_3
    n : Type u_4
    α : Type u_5
    β : Type u_6
    inst✝³ : Fintype m
    inst✝² : Fintype n
    inst✝¹ : SeminormedAddCommGroup α
    inst✝ : SeminormedAddCommGroup β
    A : Matrix m n α
    f : α → β
    hf : ∀ (a : α), Eq (NNNorm.nnnorm (f a)) (NNNorm.nnnorm a)
    ⊢ Eq (NNNorm.nnnorm (A.map f)) (NNNorm.nnnorm A)
  -/
  simp only [nnnorm_def, Pi.nnnorm_def, Matrix.map_apply, hf]
  /-
    🎉 no goals
  -/


@[simp]
theorem norm_map_eq (A : Matrix m n α) (f : α → β) (hf : ∀ a, ‖f a‖ = ‖a‖) : ‖A.map f‖ = ‖A‖ :=
  (congr_arg ((↑) : ℝ≥0 → ℝ) <| nnnorm_map_eq A f fun a => Subtype.ext <| hf a : _)


@[simp]
theorem nnnorm_transpose (A : Matrix m n α) : ‖Aᵀ‖₊ = ‖A‖₊ :=
  Finset.sup_comm _ _ _


@[simp]
theorem norm_transpose (A : Matrix m n α) : ‖Aᵀ‖ = ‖A‖ :=
  congr_arg ((↑) : ℝ≥0 → ℝ) <| nnnorm_transpose A


@[simp]
theorem nnnorm_conjTranspose [StarAddMonoid α] [NormedStarGroup α] (A : Matrix m n α) :
    ‖Aᴴ‖₊ = ‖A‖₊ :=
  (nnnorm_map_eq _ _ nnnorm_star).trans A.nnnorm_transpose


@[simp]
theorem norm_conjTranspose [StarAddMonoid α] [NormedStarGroup α] (A : Matrix m n α) : ‖Aᴴ‖ = ‖A‖ :=
  congr_arg ((↑) : ℝ≥0 → ℝ) <| nnnorm_conjTranspose A


instance [StarAddMonoid α] [NormedStarGroup α] : NormedStarGroup (Matrix m m α) :=
  ⟨norm_conjTranspose⟩


@[simp]
theorem nnnorm_col (v : m → α) : ‖col ι v‖₊ = ‖v‖₊ := by
  /-
    m : Type u_3
    α : Type u_5
    ι : Type u_7
    inst✝² : Fintype m
    inst✝¹ : Unique ι
    inst✝ : SeminormedAddCommGroup α
    v : m → α
    ⊢ Eq (NNNorm.nnnorm (Matrix.col ι v)) (NNNorm.nnnorm v)
  -/
  simp [nnnorm_def, Pi.nnnorm_def]
  /-
    🎉 no goals
  -/


@[simp]
theorem norm_col (v : m → α) : ‖col ι v‖ = ‖v‖ :=
  congr_arg ((↑) : ℝ≥0 → ℝ) <| nnnorm_col v


@[simp]
theorem nnnorm_row (v : n → α) : ‖row ι v‖₊ = ‖v‖₊ := by
  /-
    n : Type u_4
    α : Type u_5
    ι : Type u_7
    inst✝² : Fintype n
    inst✝¹ : Unique ι
    inst✝ : SeminormedAddCommGroup α
    v : n → α
    ⊢ Eq (NNNorm.nnnorm (Matrix.row ι v)) (NNNorm.nnnorm v)
  -/
  simp [nnnorm_def, Pi.nnnorm_def]
  /-
    🎉 no goals
  -/


@[simp]
theorem norm_row (v : n → α) : ‖row ι v‖ = ‖v‖ :=
  congr_arg ((↑) : ℝ≥0 → ℝ) <| nnnorm_row v


@[simp]
theorem nnnorm_diagonal [DecidableEq n] (v : n → α) : ‖diagonal v‖₊ = ‖v‖₊ := by
  /-
    n : Type u_4
    α : Type u_5
    inst✝² : Fintype n
    inst✝¹ : SeminormedAddCommGroup α
    inst✝ : DecidableEq n
    v : n → α
    ⊢ Eq (NNNorm.nnnorm (Matrix.diagonal v)) (NNNorm.nnnorm v)
  -/
  simp_rw [nnnorm_def, Pi.nnnorm_def]
  /-
    n : Type u_4
    α : Type u_5
    inst✝² : Fintype n
    inst✝¹ : SeminormedAddCommGroup α
    inst✝ : DecidableEq n
    v : n → α
    ⊢ Eq (Finset.univ.sup fun b => Finset.univ.sup fun b_1 => NNNorm.nnnorm (Matri …
  -/
  congr 1 with i : 1
  /-
    case e_f.h
    n : Type u_4
    α : Type u_5
    inst✝² : Fintype n
    inst✝¹ : SeminormedAddCommGroup α
    inst✝ : DecidableEq n
    v : n → α
    i : n
    ⊢ Eq (Finset.univ.sup fun b => NNNorm.nnnorm (Matrix.diagonal v i b)) (NNNorm. …
  -/
  refine le_antisymm (Finset.sup_le fun j hj => ?_) ?_
    /-
      case e_f.h.refine_1
      n : Type u_4
      α : Type u_5
      inst✝² : Fintype n
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : DecidableEq n
      v : n → α
      i j : n
      hj : Membership.mem Finset.univ j
      ⊢ LE.le (NNNorm.nnnorm (Matrix.diagonal v i j)) (NNNorm.nnnorm (v i))
    -/
  · obtain rfl | hij := eq_or_ne i j
      /-
        case e_f.h.refine_1.inl
        n : Type u_4
        α : Type u_5
        inst✝² : Fintype n
        inst✝¹ : SeminormedAddCommGroup α
        inst✝ : DecidableEq n
        v : n → α
        i : n
        hj : Membership.mem Finset.univ i
        ⊢ LE.le (NNNorm.nnnorm (Matrix.diagonal v i i)) (NNNorm.nnnorm (v i))
      -/
    · rw [diagonal_apply_eq]
      /-
        🎉 no goals
      -/
      /-
        case e_f.h.refine_1.inr
        n : Type u_4
        α : Type u_5
        inst✝² : Fintype n
        inst✝¹ : SeminormedAddCommGroup α
        inst✝ : DecidableEq n
        v : n → α
        i j : n
        hj : Membership.mem Finset.univ j
        hij : Ne i j
        ⊢ LE.le (NNNorm.nnnorm (Matrix.diagonal v i j)) (NNNorm.nnnorm (v i))
      -/
    · rw [diagonal_apply_ne _ hij, nnnorm_zero]
      /-
        case e_f.h.refine_1.inr
        n : Type u_4
        α : Type u_5
        inst✝² : Fintype n
        inst✝¹ : SeminormedAddCommGroup α
        inst✝ : DecidableEq n
        v : n → α
        i j : n
        hj : Membership.mem Finset.univ j
        hij : Ne i j
        ⊢ LE.le 0 (NNNorm.nnnorm (v i))
      -/
      exact zero_le _
      /-
        🎉 no goals
      -/
    /-
      case e_f.h.refine_2
      n : Type u_4
      α : Type u_5
      inst✝² : Fintype n
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : DecidableEq n
      v : n → α
      i : n
      ⊢ LE.le (NNNorm.nnnorm (v i)) (Finset.univ.sup fun b => NNNorm.nnnorm (Matrix. …
    -/
  · refine Eq.trans_le ?_ (Finset.le_sup (Finset.mem_univ i))
    /-
      case e_f.h.refine_2
      n : Type u_4
      α : Type u_5
      inst✝² : Fintype n
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : DecidableEq n
      v : n → α
      i : n
      ⊢ Eq (NNNorm.nnnorm (v i)) (NNNorm.nnnorm (Matrix.diagonal v i i))
    -/
    rw [diagonal_apply_eq]
    /-
      🎉 no goals
    -/


@[simp]
theorem norm_diagonal [DecidableEq n] (v : n → α) : ‖diagonal v‖ = ‖v‖ :=
  congr_arg ((↑) : ℝ≥0 → ℝ) <| nnnorm_diagonal v


/-- Note this is safe as an instance as it carries no data. -/
-- Porting note: not yet implemented: `@[nolint fails_quickly]`
instance [Nonempty n] [DecidableEq n] [One α] [NormOneClass α] : NormOneClass (Matrix n n α) :=
  ⟨(norm_diagonal _).trans <| norm_one⟩


/-- Normed group instance (using sup norm of sup norm) for matrices over a normed group.  Not
declared as an instance because there are several natural choices for defining the norm of a
matrix. -/
protected def normedAddCommGroup [NormedAddCommGroup α] : NormedAddCommGroup (Matrix m n α) :=
  Pi.normedAddCommGroup


/-- This applies to the sup norm of sup norm. -/
protected theorem boundedSMul [SeminormedRing R] [SeminormedAddCommGroup α] [Module R α]
    [BoundedSMul R α] : BoundedSMul R (Matrix m n α) :=
  Pi.instBoundedSMul


/-- Normed space instance (using sup norm of sup norm) for matrices over a normed space.  Not
declared as an instance because there are several natural choices for defining the norm of a
matrix. -/
protected def normedSpace : NormedSpace R (Matrix m n α) :=
  Pi.normedSpace


/-- Seminormed group instance (using sup norm of L1 norm) for matrices over a seminormed group. Not
declared as an instance because there are several natural choices for defining the norm of a
matrix. -/
@[local instance]
protected def linftyOpSeminormedAddCommGroup [SeminormedAddCommGroup α] :
    SeminormedAddCommGroup (Matrix m n α) :=
      /-
        R : Type u_1
        l : Type u_2
        m : Type u_3
        n : Type u_4
        α : Type u_5
        β : Type u_6
        ι : Type u_7
        inst✝⁴ : Fintype l
        inst✝³ : Fintype m
        inst✝² : Fintype n
        inst✝¹ : Unique ι
        inst✝ : SeminormedAddCommGroup α
        ⊢ SeminormedAddCommGroup (m → PiLp 1 fun j => α)
      -/
  (by infer_instance : SeminormedAddCommGroup (m → PiLp 1 fun j : n => α))
      /-
        🎉 no goals
      -/


/-- Normed group instance (using sup norm of L1 norm) for matrices over a normed ring.  Not
declared as an instance because there are several natural choices for defining the norm of a
matrix. -/
@[local instance]
protected def linftyOpNormedAddCommGroup [NormedAddCommGroup α] :
    NormedAddCommGroup (Matrix m n α) :=
      /-
        R : Type u_1
        l : Type u_2
        m : Type u_3
        n : Type u_4
        α : Type u_5
        β : Type u_6
        ι : Type u_7
        inst✝⁴ : Fintype l
        inst✝³ : Fintype m
        inst✝² : Fintype n
        inst✝¹ : Unique ι
        inst✝ : NormedAddCommGroup α
        ⊢ NormedAddCommGroup (m → PiLp 1 fun j => α)
      -/
  (by infer_instance : NormedAddCommGroup (m → PiLp 1 fun j : n => α))
      /-
        🎉 no goals
      -/


/-- This applies to the sup norm of L1 norm. -/
@[local instance]
protected theorem linftyOpBoundedSMul
    [SeminormedRing R] [SeminormedAddCommGroup α] [Module R α] [BoundedSMul R α] :
    BoundedSMul R (Matrix m n α) :=
      /-
        R : Type u_1
        m : Type u_3
        n : Type u_4
        α : Type u_5
        inst✝⁵ : Fintype m
        inst✝⁴ : Fintype n
        inst✝³ : SeminormedRing R
        inst✝² : SeminormedAddCommGroup α
        inst✝¹ : Module R α
        inst✝ : BoundedSMul R α
        ⊢ BoundedSMul R (m → PiLp 1 fun j => α)
      -/
  (by infer_instance : BoundedSMul R (m → PiLp 1 fun j : n => α))
      /-
        🎉 no goals
      -/


/-- Normed space instance (using sup norm of L1 norm) for matrices over a normed space.  Not
declared as an instance because there are several natural choices for defining the norm of a
matrix. -/
@[local instance]
protected def linftyOpNormedSpace [NormedField R] [SeminormedAddCommGroup α] [NormedSpace R α] :
    NormedSpace R (Matrix m n α) :=
      /-
        R : Type u_1
        l : Type u_2
        m : Type u_3
        n : Type u_4
        α : Type u_5
        β : Type u_6
        ι : Type u_7
        inst✝⁶ : Fintype l
        inst✝⁵ : Fintype m
        inst✝⁴ : Fintype n
        inst✝³ : Unique ι
        inst✝² : NormedField R
        inst✝¹ : SeminormedAddCommGroup α
        inst✝ : NormedSpace R α
        ⊢ NormedSpace R (m → PiLp 1 fun j => α)
      -/
  (by infer_instance : NormedSpace R (m → PiLp 1 fun j : n => α))
      /-
        🎉 no goals
      -/


theorem linfty_opNorm_def (A : Matrix m n α) :
    ‖A‖ = ((Finset.univ : Finset m).sup fun i : m => ∑ j : n, ‖A i j‖₊ : ℝ≥0) := by
  -- Porting note: added
  /-
    m : Type u_3
    n : Type u_4
    α : Type u_5
    inst✝² : Fintype m
    inst✝¹ : Fintype n
    inst✝ : SeminormedAddCommGroup α
    A : Matrix m n α
    ⊢ Eq (Norm.norm A) ↑(Finset.univ.sup fun i => Finset.univ.sum fun j => NNNorm. …
  -/
  change ‖fun i => (WithLp.equiv 1 _).symm (A i)‖ = _
  /-
    m : Type u_3
    n : Type u_4
    α : Type u_5
    inst✝² : Fintype m
    inst✝¹ : Fintype n
    inst✝ : SeminormedAddCommGroup α
    A : Matrix m n α
    ⊢ Eq (Norm.norm fun i => (WithLp.equiv 1 (n → α)).symm (A i)) ↑(Finset.univ.su …
  -/
  simp [Pi.norm_def, PiLp.nnnorm_eq_of_L1]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-02")] alias linfty_op_norm_def := linfty_opNorm_def


theorem linfty_opNNNorm_def (A : Matrix m n α) :
    ‖A‖₊ = (Finset.univ : Finset m).sup fun i : m => ∑ j : n, ‖A i j‖₊ :=
  Subtype.ext <| linfty_opNorm_def A


@[deprecated (since := "2024-02-02")] alias linfty_op_nnnorm_def := linfty_opNNNorm_def


@[simp]
theorem linfty_opNNNorm_col (v : m → α) : ‖col ι v‖₊ = ‖v‖₊ := by
  /-
    m : Type u_3
    α : Type u_5
    ι : Type u_7
    inst✝² : Fintype m
    inst✝¹ : Unique ι
    inst✝ : SeminormedAddCommGroup α
    v : m → α
    ⊢ Eq (NNNorm.nnnorm (Matrix.col ι v)) (NNNorm.nnnorm v)
  -/
  rw [linfty_opNNNorm_def, Pi.nnnorm_def]
  /-
    m : Type u_3
    α : Type u_5
    ι : Type u_7
    inst✝² : Fintype m
    inst✝¹ : Unique ι
    inst✝ : SeminormedAddCommGroup α
    v : m → α
    ⊢ Eq (Finset.univ.sup fun i => Finset.univ.sum fun j => NNNorm.nnnorm (Matrix. …
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-02")] alias linfty_op_nnnorm_col := linfty_opNNNorm_col


@[simp]
theorem linfty_opNorm_col (v : m → α) : ‖col ι v‖ = ‖v‖ :=
  congr_arg ((↑) : ℝ≥0 → ℝ) <| linfty_opNNNorm_col v


@[deprecated (since := "2024-02-02")] alias linfty_op_norm_col := linfty_opNorm_col


@[simp]
theorem linfty_opNNNorm_row (v : n → α) : ‖row ι v‖₊ = ∑ i, ‖v i‖₊ := by
  /-
    n : Type u_4
    α : Type u_5
    ι : Type u_7
    inst✝² : Fintype n
    inst✝¹ : Unique ι
    inst✝ : SeminormedAddCommGroup α
    v : n → α
    ⊢ Eq (NNNorm.nnnorm (Matrix.row ι v)) (Finset.univ.sum fun i => NNNorm.nnnorm  …
  -/
  simp [linfty_opNNNorm_def]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-02")] alias linfty_op_nnnorm_row := linfty_opNNNorm_row


@[simp]
theorem linfty_opNorm_row (v : n → α) : ‖row ι v‖ = ∑ i, ‖v i‖ :=
                                                                   /-
                                                                     n : Type u_4
                                                                     α : Type u_5
                                                                     ι : Type u_7
                                                                     inst✝² : Fintype n
                                                                     inst✝¹ : Unique ι
                                                                     inst✝ : SeminormedAddCommGroup α
                                                                     v : n → α
                                                                     ⊢ Eq (↑(Finset.univ.sum fun i => NNNorm.nnnorm (v i))) (Finset.univ.sum fun i  …
                                                                   -/
  (congr_arg ((↑) : ℝ≥0 → ℝ) <| linfty_opNNNorm_row v).trans <| by simp [NNReal.coe_sum]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[deprecated (since := "2024-02-02")] alias linfty_op_norm_row := linfty_opNorm_row


@[simp]
theorem linfty_opNNNorm_diagonal [DecidableEq m] (v : m → α) : ‖diagonal v‖₊ = ‖v‖₊ := by
  /-
    m : Type u_3
    α : Type u_5
    inst✝² : Fintype m
    inst✝¹ : SeminormedAddCommGroup α
    inst✝ : DecidableEq m
    v : m → α
    ⊢ Eq (NNNorm.nnnorm (Matrix.diagonal v)) (NNNorm.nnnorm v)
  -/
  rw [linfty_opNNNorm_def, Pi.nnnorm_def]
  /-
    m : Type u_3
    α : Type u_5
    inst✝² : Fintype m
    inst✝¹ : SeminormedAddCommGroup α
    inst✝ : DecidableEq m
    v : m → α
    ⊢ Eq (Finset.univ.sup fun i => Finset.univ.sum fun j => NNNorm.nnnorm (Matrix. …
  -/
  congr 1 with i : 1
  /-
    case e_f.h
    m : Type u_3
    α : Type u_5
    inst✝² : Fintype m
    inst✝¹ : SeminormedAddCommGroup α
    inst✝ : DecidableEq m
    v : m → α
    i : m
    ⊢ Eq (Finset.univ.sum fun j => NNNorm.nnnorm (Matrix.diagonal v i j)) (NNNorm. …
  -/
  refine (Finset.sum_eq_single_of_mem _ (Finset.mem_univ i) fun j _hj hij => ?_).trans ?_
    /-
      case e_f.h.refine_1
      m : Type u_3
      α : Type u_5
      inst✝² : Fintype m
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : DecidableEq m
      v : m → α
      i j : m
      _hj : Membership.mem Finset.univ j
      hij : Ne j i
      ⊢ Eq (NNNorm.nnnorm (Matrix.diagonal v i j)) 0
    -/
  · rw [diagonal_apply_ne' _ hij, nnnorm_zero]
    /-
      🎉 no goals
    -/
    /-
      case e_f.h.refine_2
      m : Type u_3
      α : Type u_5
      inst✝² : Fintype m
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : DecidableEq m
      v : m → α
      i : m
      ⊢ Eq (NNNorm.nnnorm (Matrix.diagonal v i i)) (NNNorm.nnnorm (v i))
    -/
  · rw [diagonal_apply_eq]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-02-02")] alias linfty_op_nnnorm_diagonal := linfty_opNNNorm_diagonal


@[simp]
theorem linfty_opNorm_diagonal [DecidableEq m] (v : m → α) : ‖diagonal v‖ = ‖v‖ :=
  congr_arg ((↑) : ℝ≥0 → ℝ) <| linfty_opNNNorm_diagonal v


@[deprecated (since := "2024-02-02")] alias linfty_op_norm_diagonal := linfty_opNorm_diagonal


theorem linfty_opNNNorm_mul (A : Matrix l m α) (B : Matrix m n α) : ‖A * B‖₊ ≤ ‖A‖₊ * ‖B‖₊ := by
  /-
    l : Type u_2
    m : Type u_3
    n : Type u_4
    α : Type u_5
    inst✝³ : Fintype l
    inst✝² : Fintype m
    inst✝¹ : Fintype n
    inst✝ : NonUnitalSeminormedRing α
    A : Matrix l m α
    B : Matrix m n α
    ⊢ LE.le (NNNorm.nnnorm (HMul.hMul A B)) (HMul.hMul (NNNorm.nnnorm A) (NNNorm.n …
  -/
  simp_rw [linfty_opNNNorm_def, Matrix.mul_apply]
  calc
    (Finset.univ.sup fun i => ∑ k, ‖∑ j, A i j * B j k‖₊) ≤
        Finset.univ.sup fun i => ∑ k, ∑ j, ‖A i j‖₊ * ‖B j k‖₊ :=
      Finset.sup_mono_fun fun i _hi =>
        Finset.sum_le_sum fun k _hk => nnnorm_sum_le_of_le _ fun j _hj => nnnorm_mul_le _ _
    _ = Finset.univ.sup fun i => ∑ j, ‖A i j‖₊ * ∑ k, ‖B j k‖₊ := by
      simp_rw [@Finset.sum_comm m, Finset.mul_sum]
    _ ≤ Finset.univ.sup fun i => ∑ j, ‖A i j‖₊ * Finset.univ.sup fun i => ∑ j, ‖B i j‖₊ := by
      refine Finset.sup_mono_fun fun i _hi => ?_
      gcongr with j hj
      exact Finset.le_sup (f := fun i ↦ ∑ k : n, ‖B i k‖₊) hj
    _ ≤ (Finset.univ.sup fun i => ∑ j, ‖A i j‖₊) * Finset.univ.sup fun i => ∑ j, ‖B i j‖₊ := by
      simp_rw [← Finset.sum_mul, ← NNReal.finset_sup_mul]
      rfl


@[deprecated (since := "2024-02-02")] alias linfty_op_nnnorm_mul := linfty_opNNNorm_mul


theorem linfty_opNorm_mul (A : Matrix l m α) (B : Matrix m n α) : ‖A * B‖ ≤ ‖A‖ * ‖B‖ :=
  linfty_opNNNorm_mul _ _


@[deprecated (since := "2024-02-02")] alias linfty_op_norm_mul := linfty_opNorm_mul


theorem linfty_opNNNorm_mulVec (A : Matrix l m α) (v : m → α) : ‖A *ᵥ v‖₊ ≤ ‖A‖₊ * ‖v‖₊ := by
  /-
    l : Type u_2
    m : Type u_3
    α : Type u_5
    inst✝² : Fintype l
    inst✝¹ : Fintype m
    inst✝ : NonUnitalSeminormedRing α
    A : Matrix l m α
    v : m → α
    ⊢ LE.le (NNNorm.nnnorm (A.mulVec v)) (HMul.hMul (NNNorm.nnnorm A) (NNNorm.nnno …
  -/
  rw [← linfty_opNNNorm_col (ι := Fin 1) (A *ᵥ v), ← linfty_opNNNorm_col v (ι := Fin 1)]
  /-
    l : Type u_2
    m : Type u_3
    α : Type u_5
    inst✝² : Fintype l
    inst✝¹ : Fintype m
    inst✝ : NonUnitalSeminormedRing α
    A : Matrix l m α
    v : m → α
    ⊢ LE.le (NNNorm.nnnorm (Matrix.col (Fin 1) (A.mulVec v))) (HMul.hMul (NNNorm.n …
  -/
  exact linfty_opNNNorm_mul A (col (Fin 1) v)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-02")] alias linfty_op_nnnorm_mulVec := linfty_opNNNorm_mulVec


theorem linfty_opNorm_mulVec (A : Matrix l m α) (v : m → α) : ‖A *ᵥ v‖ ≤ ‖A‖ * ‖v‖ :=
  linfty_opNNNorm_mulVec _ _


@[deprecated (since := "2024-02-02")] alias linfty_op_norm_mulVec := linfty_opNorm_mulVec


/-- Seminormed non-unital ring instance (using sup norm of L1 norm) for matrices over a semi normed
non-unital ring. Not declared as an instance because there are several natural choices for defining
the norm of a matrix. -/
@[local instance]
protected def linftyOpNonUnitalSemiNormedRing [NonUnitalSeminormedRing α] :
    NonUnitalSeminormedRing (Matrix n n α) :=
  { Matrix.linftyOpSeminormedAddCommGroup, Matrix.instNonUnitalRing with
    norm_mul := linfty_opNorm_mul }


/-- The `L₁-L∞` norm preserves one on non-empty matrices. Note this is safe as an instance, as it
carries no data. -/
instance linfty_opNormOneClass [SeminormedRing α] [NormOneClass α] [DecidableEq n] [Nonempty n] :
    NormOneClass (Matrix n n α) where norm_one := (linfty_opNorm_diagonal _).trans norm_one


/-- Seminormed ring instance (using sup norm of L1 norm) for matrices over a semi normed ring.  Not
declared as an instance because there are several natural choices for defining the norm of a
matrix. -/
@[local instance]
protected def linftyOpSemiNormedRing [SeminormedRing α] [DecidableEq n] :
    SeminormedRing (Matrix n n α) :=
  { Matrix.linftyOpNonUnitalSemiNormedRing, Matrix.instRing with }


/-- Normed non-unital ring instance (using sup norm of L1 norm) for matrices over a normed
non-unital ring. Not declared as an instance because there are several natural choices for defining
the norm of a matrix. -/
@[local instance]
protected def linftyOpNonUnitalNormedRing [NonUnitalNormedRing α] :
    NonUnitalNormedRing (Matrix n n α) :=
  { Matrix.linftyOpNonUnitalSemiNormedRing with
    eq_of_dist_eq_zero := eq_of_dist_eq_zero }


/-- Normed ring instance (using sup norm of L1 norm) for matrices over a normed ring.  Not
declared as an instance because there are several natural choices for defining the norm of a
matrix. -/
@[local instance]
protected def linftyOpNormedRing [NormedRing α] [DecidableEq n] : NormedRing (Matrix n n α) :=
  { Matrix.linftyOpSemiNormedRing with
    eq_of_dist_eq_zero := eq_of_dist_eq_zero }


/-- Normed algebra instance (using sup norm of L1 norm) for matrices over a normed algebra. Not
declared as an instance because there are several natural choices for defining the norm of a
matrix. -/
@[local instance]
protected def linftyOpNormedAlgebra [NormedField R] [SeminormedRing α] [NormedAlgebra R α]
    [DecidableEq n] : NormedAlgebra R (Matrix n n α) :=
  { Matrix.linftyOpNormedSpace, Matrix.instAlgebra with }



/-- Auxiliary construction; an element of norm 1 such that `a * unitOf a = ‖a‖`. -/
                                     /-
                                       R : Type u_1
                                       l : Type u_2
                                       m : Type u_3
                                       n : Type u_4
                                       α : Type u_5
                                       β : Type u_6
                                       ι : Type u_7
                                       inst✝⁵ : Fintype l
                                       inst✝⁴ : Fintype m
                                       inst✝³ : Fintype n
                                       inst✝² : Unique ι
                                       inst✝¹ : NormedDivisionRing α
                                       inst✝ : NormedAlgebra Real α
                                       a : α
                                       ⊢ α
                                     -/
private def unitOf (a : α) : α := by classical exact if a = 0 then 1 else ‖a‖ • a⁻¹
                                     /-
                                       🎉 no goals
                                     -/


private theorem norm_unitOf (a : α) : ‖unitOf a‖₊ = 1 := by
  /-
    α : Type u_5
    inst✝¹ : NormedDivisionRing α
    inst✝ : NormedAlgebra Real α
    a : α
    ⊢ Eq (NNNorm.nnnorm (Matrix.unitOf a)) 1
  -/
  rw [unitOf]
  /-
    α : Type u_5
    inst✝¹ : NormedDivisionRing α
    inst✝ : NormedAlgebra Real α
    a : α
    ⊢ Eq (NNNorm.nnnorm (ite (Eq a 0) 1 (HSMul.hSMul (Norm.norm a) (Inv.inv a)))) 1
  -/
  split_ifs with h
    /-
      case pos
      α : Type u_5
      inst✝¹ : NormedDivisionRing α
      inst✝ : NormedAlgebra Real α
      a : α
      h : Eq a 0
      ⊢ Eq (NNNorm.nnnorm 1) 1
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_5
      inst✝¹ : NormedDivisionRing α
      inst✝ : NormedAlgebra Real α
      a : α
      h : Not (Eq a 0)
      ⊢ Eq (NNNorm.nnnorm (HSMul.hSMul (Norm.norm a) (Inv.inv a))) 1
    -/
  · rw [← nnnorm_eq_zero] at h
    /-
      case neg
      α : Type u_5
      inst✝¹ : NormedDivisionRing α
      inst✝ : NormedAlgebra Real α
      a : α
      h : Not (Eq (NNNorm.nnnorm a) 0)
      ⊢ Eq (NNNorm.nnnorm (HSMul.hSMul (Norm.norm a) (Inv.inv a))) 1
    -/
    rw [nnnorm_smul, nnnorm_inv, nnnorm_norm, mul_inv_cancel₀ h]
    /-
      🎉 no goals
    -/


private theorem mul_unitOf (a : α) : a * unitOf a = algebraMap _ _ (‖a‖₊ : ℝ)  := by
  /-
    α : Type u_5
    inst✝¹ : NormedDivisionRing α
    inst✝ : NormedAlgebra Real α
    a : α
    ⊢ Eq (HMul.hMul a (Matrix.unitOf a)) ((algebraMap Real α) ↑(NNNorm.nnnorm a))
  -/
  simp only [unitOf, coe_nnnorm]
  /-
    α : Type u_5
    inst✝¹ : NormedDivisionRing α
    inst✝ : NormedAlgebra Real α
    a : α
    ⊢ Eq (HMul.hMul a (ite (Eq a 0) 1 (HSMul.hSMul (Norm.norm a) (Inv.inv a)))) (( …
  -/
  split_ifs with h
    /-
      case pos
      α : Type u_5
      inst✝¹ : NormedDivisionRing α
      inst✝ : NormedAlgebra Real α
      a : α
      h : Eq a 0
      ⊢ Eq (HMul.hMul a 1) ((algebraMap Real α) (Norm.norm a))
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_5
      inst✝¹ : NormedDivisionRing α
      inst✝ : NormedAlgebra Real α
      a : α
      h : Not (Eq a 0)
      ⊢ Eq (HMul.hMul a (HSMul.hSMul (Norm.norm a) (Inv.inv a))) ((algebraMap Real α …
    -/
  · rw [mul_smul_comm, mul_inv_cancel₀ h, Algebra.algebraMap_eq_smul_one]
    /-
      🎉 no goals
    -/


lemma linfty_opNNNorm_eq_opNNNorm (A : Matrix m n α) :
            /-
              R : Type u_1
              l : Type u_2
              m : Type u_3
              n : Type u_4
              α : Type u_5
              β : Type u_6
              ι : Type u_7
              inst✝⁵ : Fintype l
              inst✝⁴ : Fintype m
              inst✝³ : Fintype n
              inst✝² : Unique ι
              inst✝¹ : NontriviallyNormedField α
              inst✝ : NormedAlgebra Real α
              A : Matrix m n α
              ⊢ Continuous A.mulVecLin.toFun
            -/
    ‖A‖₊ = ‖ContinuousLinearMap.mk (Matrix.mulVecLin A)‖₊ := by
            /-
              🎉 no goals
            -/
  /-
    m : Type u_3
    n : Type u_4
    α : Type u_5
    inst✝³ : Fintype m
    inst✝² : Fintype n
    inst✝¹ : NontriviallyNormedField α
    inst✝ : NormedAlgebra Real α
    A : Matrix m n α
    ⊢ Eq (NNNorm.nnnorm A) (NNNorm.nnnorm { toLinearMap := A.mulVecLin, cont := ⋯ })
  -/
  rw [ContinuousLinearMap.opNNNorm_eq_of_bounds _ (linfty_opNNNorm_mulVec _) fun N hN => ?_]
  /-
    m : Type u_3
    n : Type u_4
    α : Type u_5
    inst✝³ : Fintype m
    inst✝² : Fintype n
    inst✝¹ : NontriviallyNormedField α
    inst✝ : NormedAlgebra Real α
    A : Matrix m n α
    N : NNReal
    hN : ∀ (x : n → α), LE.le (NNNorm.nnnorm ({ toLinearMap := A.mulVecLin, cont : …
    ⊢ LE.le (NNNorm.nnnorm A) N
  -/
  rw [linfty_opNNNorm_def]
  /-
    m : Type u_3
    n : Type u_4
    α : Type u_5
    inst✝³ : Fintype m
    inst✝² : Fintype n
    inst✝¹ : NontriviallyNormedField α
    inst✝ : NormedAlgebra Real α
    A : Matrix m n α
    N : NNReal
    hN : ∀ (x : n → α), LE.le (NNNorm.nnnorm ({ toLinearMap := A.mulVecLin, cont : …
    ⊢ LE.le (Finset.univ.sup fun i => Finset.univ.sum fun j => NNNorm.nnnorm (A i  …
  -/
  refine Finset.sup_le fun i _ => ?_
  /-
    m : Type u_3
    n : Type u_4
    α : Type u_5
    inst✝³ : Fintype m
    inst✝² : Fintype n
    inst✝¹ : NontriviallyNormedField α
    inst✝ : NormedAlgebra Real α
    A : Matrix m n α
    N : NNReal
    hN : ∀ (x : n → α), LE.le (NNNorm.nnnorm ({ toLinearMap := A.mulVecLin, cont : …
    i : m
    x✝ : Membership.mem Finset.univ i
    ⊢ LE.le (Finset.univ.sum fun j => NNNorm.nnnorm (A i j)) N
  -/
  cases isEmpty_or_nonempty n
    /-
      case inl
      m : Type u_3
      n : Type u_4
      α : Type u_5
      inst✝³ : Fintype m
      inst✝² : Fintype n
      inst✝¹ : NontriviallyNormedField α
      inst✝ : NormedAlgebra Real α
      A : Matrix m n α
      N : NNReal
      hN : ∀ (x : n → α), LE.le (NNNorm.nnnorm ({ toLinearMap := A.mulVecLin, cont : …
      i : m
      x✝ : Membership.mem Finset.univ i
      h✝ : IsEmpty n
      ⊢ LE.le (Finset.univ.sum fun j => NNNorm.nnnorm (A i j)) N
    -/
  · simp
    /-
      🎉 no goals
    -/
  classical
  let x : n → α := fun j => unitOf (A i j)
  have hxn : ‖x‖₊ = 1 := by
    simp_rw [x, Pi.nnnorm_def, norm_unitOf, Finset.sup_const Finset.univ_nonempty]
  specialize hN x
  rw [hxn, mul_one, Pi.nnnorm_def, Finset.sup_le_iff] at hN
  replace hN := hN i (Finset.mem_univ _)
  dsimp [mulVec, dotProduct] at hN
  simp_rw [x, mul_unitOf, ← map_sum, nnnorm_algebraMap, ← NNReal.coe_sum, NNReal.nnnorm_eq,
    nnnorm_one, mul_one] at hN
  exact hN


@[deprecated (since := "2024-02-02")]
alias linfty_op_nnnorm_eq_op_nnnorm := linfty_opNNNorm_eq_opNNNorm


lemma linfty_opNorm_eq_opNorm (A : Matrix m n α) :
           /-
             R : Type u_1
             l : Type u_2
             m : Type u_3
             n : Type u_4
             α : Type u_5
             β : Type u_6
             ι : Type u_7
             inst✝⁵ : Fintype l
             inst✝⁴ : Fintype m
             inst✝³ : Fintype n
             inst✝² : Unique ι
             inst✝¹ : NontriviallyNormedField α
             inst✝ : NormedAlgebra Real α
             A : Matrix m n α
             ⊢ Continuous A.mulVecLin.toFun
           -/
    ‖A‖ = ‖ContinuousLinearMap.mk (Matrix.mulVecLin A)‖ :=
           /-
             🎉 no goals
           -/
  congr_arg NNReal.toReal (linfty_opNNNorm_eq_opNNNorm A)


@[deprecated (since := "2024-02-02")] alias linfty_op_norm_eq_op_norm := linfty_opNorm_eq_opNorm


@[simp] lemma linfty_opNNNorm_toMatrix (f : (n → α) →L[α] (m → α)) :
    ‖LinearMap.toMatrix' (↑f : (n → α) →ₗ[α] (m → α))‖₊ = ‖f‖₊ := by
  /-
    m : Type u_3
    n : Type u_4
    α : Type u_5
    inst✝⁴ : Fintype m
    inst✝³ : Fintype n
    inst✝² : NontriviallyNormedField α
    inst✝¹ : NormedAlgebra Real α
    inst✝ : DecidableEq n
    f : ContinuousLinearMap (RingHom.id α) (n → α) (m → α)
    ⊢ Eq (NNNorm.nnnorm (LinearMap.toMatrix' ↑f)) (NNNorm.nnnorm f)
  -/
  rw [linfty_opNNNorm_eq_opNNNorm]
  /-
    m : Type u_3
    n : Type u_4
    α : Type u_5
    inst✝⁴ : Fintype m
    inst✝³ : Fintype n
    inst✝² : NontriviallyNormedField α
    inst✝¹ : NormedAlgebra Real α
    inst✝ : DecidableEq n
    f : ContinuousLinearMap (RingHom.id α) (n → α) (m → α)
    ⊢ Eq (NNNorm.nnnorm { toLinearMap := (LinearMap.toMatrix' ↑f).mulVecLin, cont  …
  -/
  simp only [← toLin'_apply', toLin'_toMatrix']
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-02")] alias linfty_op_nnnorm_toMatrix := linfty_opNNNorm_toMatrix


@[simp] lemma linfty_opNorm_toMatrix (f : (n → α) →L[α] (m → α)) :
    ‖LinearMap.toMatrix' (↑f : (n → α) →ₗ[α] (m → α))‖ = ‖f‖ :=
  congr_arg NNReal.toReal (linfty_opNNNorm_toMatrix f)


@[deprecated (since := "2024-02-02")] alias linfty_op_norm_toMatrix := linfty_opNorm_toMatrix


/-- Seminormed group instance (using frobenius norm) for matrices over a seminormed group. Not
declared as an instance because there are several natural choices for defining the norm of a
matrix. -/
@[local instance]
def frobeniusSeminormedAddCommGroup [SeminormedAddCommGroup α] :
    SeminormedAddCommGroup (Matrix m n α) :=
  inferInstanceAs (SeminormedAddCommGroup (PiLp 2 fun _i : m => PiLp 2 fun _j : n => α))


/-- Normed group instance (using frobenius norm) for matrices over a normed group.  Not
declared as an instance because there are several natural choices for defining the norm of a
matrix. -/
@[local instance]
def frobeniusNormedAddCommGroup [NormedAddCommGroup α] : NormedAddCommGroup (Matrix m n α) :=
      /-
        R : Type u_1
        l : Type u_2
        m : Type u_3
        n : Type u_4
        α : Type u_5
        β : Type u_6
        ι : Type u_7
        inst✝⁴ : Fintype l
        inst✝³ : Fintype m
        inst✝² : Fintype n
        inst✝¹ : Unique ι
        inst✝ : NormedAddCommGroup α
        ⊢ NormedAddCommGroup (PiLp 2 fun i => PiLp 2 fun j => α)
      -/
  (by infer_instance : NormedAddCommGroup (PiLp 2 fun i : m => PiLp 2 fun j : n => α))
      /-
        🎉 no goals
      -/


/-- This applies to the frobenius norm. -/
@[local instance]
theorem frobeniusBoundedSMul [SeminormedRing R] [SeminormedAddCommGroup α] [Module R α]
    [BoundedSMul R α] :
    BoundedSMul R (Matrix m n α) :=
      /-
        R : Type u_1
        m : Type u_3
        n : Type u_4
        α : Type u_5
        inst✝⁵ : Fintype m
        inst✝⁴ : Fintype n
        inst✝³ : SeminormedRing R
        inst✝² : SeminormedAddCommGroup α
        inst✝¹ : Module R α
        inst✝ : BoundedSMul R α
        ⊢ BoundedSMul R (PiLp 2 fun i => PiLp 2 fun j => α)
      -/
  (by infer_instance : BoundedSMul R (PiLp 2 fun i : m => PiLp 2 fun j : n => α))
      /-
        🎉 no goals
      -/


/-- Normed space instance (using frobenius norm) for matrices over a normed space.  Not
declared as an instance because there are several natural choices for defining the norm of a
matrix. -/
@[local instance]
def frobeniusNormedSpace [NormedField R] [SeminormedAddCommGroup α] [NormedSpace R α] :
    NormedSpace R (Matrix m n α) :=
      /-
        R : Type u_1
        l : Type u_2
        m : Type u_3
        n : Type u_4
        α : Type u_5
        β : Type u_6
        ι : Type u_7
        inst✝⁶ : Fintype l
        inst✝⁵ : Fintype m
        inst✝⁴ : Fintype n
        inst✝³ : Unique ι
        inst✝² : NormedField R
        inst✝¹ : SeminormedAddCommGroup α
        inst✝ : NormedSpace R α
        ⊢ NormedSpace R (PiLp 2 fun i => PiLp 2 fun j => α)
      -/
  (by infer_instance : NormedSpace R (PiLp 2 fun i : m => PiLp 2 fun j : n => α))
      /-
        🎉 no goals
      -/


theorem frobenius_nnnorm_def (A : Matrix m n α) :
    ‖A‖₊ = (∑ i, ∑ j, ‖A i j‖₊ ^ (2 : ℝ)) ^ (1 / 2 : ℝ) := by
  -- Porting note: added, along with `WithLp.equiv_symm_pi_apply` below
  /-
    m : Type u_3
    n : Type u_4
    α : Type u_5
    inst✝² : Fintype m
    inst✝¹ : Fintype n
    inst✝ : SeminormedAddCommGroup α
    A : Matrix m n α
    ⊢ Eq (NNNorm.nnnorm A) (HPow.hPow (Finset.univ.sum fun i => Finset.univ.sum fu …
  -/
  change ‖(WithLp.equiv 2 _).symm fun i => (WithLp.equiv 2 _).symm fun j => A i j‖₊ = _
  simp_rw [PiLp.nnnorm_eq_of_L2, NNReal.sq_sqrt, NNReal.sqrt_eq_rpow, NNReal.rpow_two,
    WithLp.equiv_symm_pi_apply]


theorem frobenius_norm_def (A : Matrix m n α) :
    ‖A‖ = (∑ i, ∑ j, ‖A i j‖ ^ (2 : ℝ)) ^ (1 / 2 : ℝ) :=
                                                                   /-
                                                                     m : Type u_3
                                                                     n : Type u_4
                                                                     α : Type u_5
                                                                     inst✝² : Fintype m
                                                                     inst✝¹ : Fintype n
                                                                     inst✝ : SeminormedAddCommGroup α
                                                                     A : Matrix m n α
                                                                     ⊢ Eq (↑(HPow.hPow (Finset.univ.sum fun i => Finset.univ.sum fun j => HPow.hPow …
                                                                   -/
  (congr_arg ((↑) : ℝ≥0 → ℝ) (frobenius_nnnorm_def A)).trans <| by simp [NNReal.coe_sum]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
theorem frobenius_nnnorm_map_eq (A : Matrix m n α) (f : α → β) (hf : ∀ a, ‖f a‖₊ = ‖a‖₊) :
                            /-
                              m : Type u_3
                              n : Type u_4
                              α : Type u_5
                              β : Type u_6
                              inst✝³ : Fintype m
                              inst✝² : Fintype n
                              inst✝¹ : SeminormedAddCommGroup α
                              inst✝ : SeminormedAddCommGroup β
                              A : Matrix m n α
                              f : α → β
                              hf : ∀ (a : α), Eq (NNNorm.nnnorm (f a)) (NNNorm.nnnorm a)
                              ⊢ Eq (NNNorm.nnnorm (A.map f)) (NNNorm.nnnorm A)
                            -/
    ‖A.map f‖₊ = ‖A‖₊ := by simp_rw [frobenius_nnnorm_def, Matrix.map_apply, hf]
                            /-
                              🎉 no goals
                            -/


@[simp]
theorem frobenius_norm_map_eq (A : Matrix m n α) (f : α → β) (hf : ∀ a, ‖f a‖ = ‖a‖) :
    ‖A.map f‖ = ‖A‖ :=
  (congr_arg ((↑) : ℝ≥0 → ℝ) <| frobenius_nnnorm_map_eq A f fun a => Subtype.ext <| hf a : _)


@[simp]
theorem frobenius_nnnorm_transpose (A : Matrix m n α) : ‖Aᵀ‖₊ = ‖A‖₊ := by
  /-
    m : Type u_3
    n : Type u_4
    α : Type u_5
    inst✝² : Fintype m
    inst✝¹ : Fintype n
    inst✝ : SeminormedAddCommGroup α
    A : Matrix m n α
    ⊢ Eq (NNNorm.nnnorm A.transpose) (NNNorm.nnnorm A)
  -/
  rw [frobenius_nnnorm_def, frobenius_nnnorm_def, Finset.sum_comm]
  /-
    m : Type u_3
    n : Type u_4
    α : Type u_5
    inst✝² : Fintype m
    inst✝¹ : Fintype n
    inst✝ : SeminormedAddCommGroup α
    A : Matrix m n α
    ⊢ Eq (HPow.hPow (Finset.univ.sum fun y => Finset.univ.sum fun x => HPow.hPow ( …
  -/
  simp_rw [Matrix.transpose_apply]  -- Porting note: added
  /-
    🎉 no goals
  -/


@[simp]
theorem frobenius_norm_transpose (A : Matrix m n α) : ‖Aᵀ‖ = ‖A‖ :=
  congr_arg ((↑) : ℝ≥0 → ℝ) <| frobenius_nnnorm_transpose A


@[simp]
theorem frobenius_nnnorm_conjTranspose [StarAddMonoid α] [NormedStarGroup α] (A : Matrix m n α) :
    ‖Aᴴ‖₊ = ‖A‖₊ :=
  (frobenius_nnnorm_map_eq _ _ nnnorm_star).trans A.frobenius_nnnorm_transpose


@[simp]
theorem frobenius_norm_conjTranspose [StarAddMonoid α] [NormedStarGroup α] (A : Matrix m n α) :
    ‖Aᴴ‖ = ‖A‖ :=
  congr_arg ((↑) : ℝ≥0 → ℝ) <| frobenius_nnnorm_conjTranspose A


instance frobenius_normedStarGroup [StarAddMonoid α] [NormedStarGroup α] :
    NormedStarGroup (Matrix m m α) :=
  ⟨frobenius_norm_conjTranspose⟩


@[simp]
theorem frobenius_norm_row (v : m → α) : ‖row ι v‖ = ‖(WithLp.equiv 2 _).symm v‖ := by
  /-
    m : Type u_3
    α : Type u_5
    ι : Type u_7
    inst✝² : Fintype m
    inst✝¹ : Unique ι
    inst✝ : SeminormedAddCommGroup α
    v : m → α
    ⊢ Eq (Norm.norm (Matrix.row ι v)) (Norm.norm ((WithLp.equiv 2 (m → α)).symm v))
  -/
  rw [frobenius_norm_def, Fintype.sum_unique, PiLp.norm_eq_of_L2, Real.sqrt_eq_rpow]
  /-
    m : Type u_3
    α : Type u_5
    ι : Type u_7
    inst✝² : Fintype m
    inst✝¹ : Unique ι
    inst✝ : SeminormedAddCommGroup α
    v : m → α
    ⊢ Eq (HPow.hPow (Finset.univ.sum fun j => HPow.hPow (Norm.norm (Matrix.row ι v …
  -/
  simp only [row_apply, Real.rpow_two, WithLp.equiv_symm_pi_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem frobenius_nnnorm_row (v : m → α) : ‖row ι v‖₊ = ‖(WithLp.equiv 2 _).symm v‖₊ :=
  Subtype.ext <| frobenius_norm_row v


@[simp]
theorem frobenius_norm_col (v : n → α) : ‖col ι v‖ = ‖(WithLp.equiv 2 _).symm v‖ := by
  /-
    n : Type u_4
    α : Type u_5
    ι : Type u_7
    inst✝² : Fintype n
    inst✝¹ : Unique ι
    inst✝ : SeminormedAddCommGroup α
    v : n → α
    ⊢ Eq (Norm.norm (Matrix.col ι v)) (Norm.norm ((WithLp.equiv 2 (n → α)).symm v))
  -/
  simp_rw [frobenius_norm_def, Fintype.sum_unique, PiLp.norm_eq_of_L2, Real.sqrt_eq_rpow]
  /-
    n : Type u_4
    α : Type u_5
    ι : Type u_7
    inst✝² : Fintype n
    inst✝¹ : Unique ι
    inst✝ : SeminormedAddCommGroup α
    v : n → α
    ⊢ Eq (HPow.hPow (Finset.univ.sum fun x => HPow.hPow (Norm.norm (Matrix.col ι v …
  -/
  simp only [col_apply, Real.rpow_two, WithLp.equiv_symm_pi_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem frobenius_nnnorm_col (v : n → α) : ‖col ι v‖₊ = ‖(WithLp.equiv 2 _).symm v‖₊ :=
  Subtype.ext <| frobenius_norm_col v


@[simp]
theorem frobenius_nnnorm_diagonal [DecidableEq n] (v : n → α) :
    ‖diagonal v‖₊ = ‖(WithLp.equiv 2 _).symm v‖₊ := by
  simp_rw [frobenius_nnnorm_def, ← Finset.sum_product', Finset.univ_product_univ,
    PiLp.nnnorm_eq_of_L2]
  /-
    n : Type u_4
    α : Type u_5
    inst✝² : Fintype n
    inst✝¹ : SeminormedAddCommGroup α
    inst✝ : DecidableEq n
    v : n → α
    ⊢ Eq (HPow.hPow (Finset.univ.sum fun x => HPow.hPow (NNNorm.nnnorm (Matrix.dia …
  -/
  let s := (Finset.univ : Finset n).map ⟨fun i : n => (i, i), fun i j h => congr_arg Prod.fst h⟩
  /-
    n : Type u_4
    α : Type u_5
    inst✝² : Fintype n
    inst✝¹ : SeminormedAddCommGroup α
    inst✝ : DecidableEq n
    v : n → α
    s : Finset (Prod n n) := Finset.map { toFun := fun i => { fst := i, snd := i } …
    ⊢ Eq (HPow.hPow (Finset.univ.sum fun x => HPow.hPow (NNNorm.nnnorm (Matrix.dia …
  -/
  rw [← Finset.sum_subset (Finset.subset_univ s) fun i _hi his => ?_]
    /-
      n : Type u_4
      α : Type u_5
      inst✝² : Fintype n
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : DecidableEq n
      v : n → α
      s : Finset (Prod n n) := Finset.map { toFun := fun i => { fst := i, snd := i } …
      ⊢ Eq (HPow.hPow (s.sum fun x => HPow.hPow (NNNorm.nnnorm (Matrix.diagonal v x. …
    -/
  · rw [Finset.sum_map, NNReal.sqrt_eq_rpow]
    /-
      n : Type u_4
      α : Type u_5
      inst✝² : Fintype n
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : DecidableEq n
      v : n → α
      s : Finset (Prod n n) := Finset.map { toFun := fun i => { fst := i, snd := i } …
      ⊢ Eq (HPow.hPow (Finset.univ.sum fun x => HPow.hPow (NNNorm.nnnorm (Matrix.dia …
    -/
    dsimp
    /-
      n : Type u_4
      α : Type u_5
      inst✝² : Fintype n
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : DecidableEq n
      v : n → α
      s : Finset (Prod n n) := Finset.map { toFun := fun i => { fst := i, snd := i } …
      ⊢ Eq (HPow.hPow (Finset.univ.sum fun x => HPow.hPow (NNNorm.nnnorm (Matrix.dia …
    -/
    simp_rw [diagonal_apply_eq, NNReal.rpow_two]
    /-
      🎉 no goals
    -/
    /-
      n : Type u_4
      α : Type u_5
      inst✝² : Fintype n
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : DecidableEq n
      v : n → α
      s : Finset (Prod n n) := Finset.map { toFun := fun i => { fst := i, snd := i } …
      i : Prod n n
      _hi : Membership.mem Finset.univ i
      his : Not (Membership.mem s i)
      ⊢ Eq (HPow.hPow (NNNorm.nnnorm (Matrix.diagonal v i.1 i.2)) 2) 0
    -/
  · suffices i.1 ≠ i.2 by rw [diagonal_apply_ne _ this, nnnorm_zero, NNReal.zero_rpow two_ne_zero]
    /-
      n : Type u_4
      α : Type u_5
      inst✝² : Fintype n
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : DecidableEq n
      v : n → α
      s : Finset (Prod n n) := Finset.map { toFun := fun i => { fst := i, snd := i } …
      i : Prod n n
      _hi : Membership.mem Finset.univ i
      his : Not (Membership.mem s i)
      ⊢ Ne i.1 i.2
    -/
    intro h
    /-
      n : Type u_4
      α : Type u_5
      inst✝² : Fintype n
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : DecidableEq n
      v : n → α
      s : Finset (Prod n n) := Finset.map { toFun := fun i => { fst := i, snd := i } …
      i : Prod n n
      _hi : Membership.mem Finset.univ i
      his : Not (Membership.mem s i)
      h : Eq i.1 i.2
      ⊢ False
    -/
    exact Finset.mem_map.not.mp his ⟨i.1, Finset.mem_univ _, Prod.ext rfl h⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem frobenius_norm_diagonal [DecidableEq n] (v : n → α) :
    ‖diagonal v‖ = ‖(WithLp.equiv 2 _).symm v‖ :=
  (congr_arg ((↑) : ℝ≥0 → ℝ) <| frobenius_nnnorm_diagonal v : _).trans rfl


theorem frobenius_nnnorm_one [DecidableEq n] [SeminormedAddCommGroup α] [One α] :
    ‖(1 : Matrix n n α)‖₊ = NNReal.sqrt (Fintype.card n) * ‖(1 : α)‖₊ := by
  /-
    n : Type u_4
    α : Type u_5
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    inst✝¹ : SeminormedAddCommGroup α
    inst✝ : One α
    ⊢ Eq (NNNorm.nnnorm 1) (HMul.hMul (NNReal.sqrt ↑(Fintype.card n)) (NNNorm.nnno …
  -/
  refine (frobenius_nnnorm_diagonal _).trans ?_
  -- Porting note: change to erw, since `fun x => 1` no longer matches `Function.const`
  /-
    n : Type u_4
    α : Type u_5
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    inst✝¹ : SeminormedAddCommGroup α
    inst✝ : One α
    ⊢ Eq (NNNorm.nnnorm ((WithLp.equiv 2 (n → α)).symm fun x => 1)) (HMul.hMul (NN …
  -/
  erw [PiLp.nnnorm_equiv_symm_const ENNReal.two_ne_top]
  /-
    n : Type u_4
    α : Type u_5
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    inst✝¹ : SeminormedAddCommGroup α
    inst✝ : One α
    ⊢ Eq (HMul.hMul (HPow.hPow (↑(Fintype.card n)) (1 / 2).toReal) (NNNorm.nnnorm  …
  -/
  simp_rw [NNReal.sqrt_eq_rpow]
  -- Porting note: added `ENNReal.toReal_ofNat`
  /-
    n : Type u_4
    α : Type u_5
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    inst✝¹ : SeminormedAddCommGroup α
    inst✝ : One α
    ⊢ Eq (HMul.hMul (HPow.hPow (↑(Fintype.card n)) (1 / 2).toReal) (NNNorm.nnnorm  …
  -/
  simp only [ENNReal.toReal_div, ENNReal.one_toReal, ENNReal.toReal_ofNat]
  /-
    🎉 no goals
  -/


theorem frobenius_nnnorm_mul (A : Matrix l m α) (B : Matrix m n α) : ‖A * B‖₊ ≤ ‖A‖₊ * ‖B‖₊ := by
  /-
    l : Type u_2
    m : Type u_3
    n : Type u_4
    α : Type u_5
    inst✝³ : Fintype l
    inst✝² : Fintype m
    inst✝¹ : Fintype n
    inst✝ : RCLike α
    A : Matrix l m α
    B : Matrix m n α
    ⊢ LE.le (NNNorm.nnnorm (HMul.hMul A B)) (HMul.hMul (NNNorm.nnnorm A) (NNNorm.n …
  -/
  simp_rw [frobenius_nnnorm_def, Matrix.mul_apply]
  /-
    l : Type u_2
    m : Type u_3
    n : Type u_4
    α : Type u_5
    inst✝³ : Fintype l
    inst✝² : Fintype m
    inst✝¹ : Fintype n
    inst✝ : RCLike α
    A : Matrix l m α
    B : Matrix m n α
    ⊢ LE.le (HPow.hPow (Finset.univ.sum fun x => Finset.univ.sum fun x_1 => HPow.h …
  -/
  rw [← NNReal.mul_rpow, @Finset.sum_comm _ _ m, Finset.sum_mul_sum]
  /-
    l : Type u_2
    m : Type u_3
    n : Type u_4
    α : Type u_5
    inst✝³ : Fintype l
    inst✝² : Fintype m
    inst✝¹ : Fintype n
    inst✝ : RCLike α
    A : Matrix l m α
    B : Matrix m n α
    ⊢ LE.le (HPow.hPow (Finset.univ.sum fun x => Finset.univ.sum fun x_1 => HPow.h …
  -/
  gcongr with i _ j
  rw [← NNReal.rpow_le_rpow_iff one_half_pos, ← NNReal.rpow_mul,
    mul_div_cancel₀ (1 : ℝ) two_ne_zero, NNReal.rpow_one, NNReal.mul_rpow]
  have :=
    @nnnorm_inner_le_nnnorm α _ _ _ _ ((WithLp.equiv 2 <| _ → α).symm fun j => star (A i j))
      ((WithLp.equiv 2 <| _ → α).symm fun k => B k j)
  simpa only [WithLp.equiv_symm_pi_apply, PiLp.inner_apply, RCLike.inner_apply, starRingEnd_apply,
    Pi.nnnorm_def, PiLp.nnnorm_eq_of_L2, star_star, nnnorm_star, NNReal.sqrt_eq_rpow,
    NNReal.rpow_two] using this


theorem frobenius_norm_mul (A : Matrix l m α) (B : Matrix m n α) : ‖A * B‖ ≤ ‖A‖ * ‖B‖ :=
  frobenius_nnnorm_mul A B


/-- Normed ring instance (using frobenius norm) for matrices over `ℝ` or `ℂ`.  Not
declared as an instance because there are several natural choices for defining the norm of a
matrix. -/
@[local instance]
def frobeniusNormedRing [DecidableEq m] : NormedRing (Matrix m m α) :=
  { Matrix.frobeniusSeminormedAddCommGroup, Matrix.instRing with
    norm := Norm.norm
    norm_mul := frobenius_norm_mul
    eq_of_dist_eq_zero := eq_of_dist_eq_zero }


/-- Normed algebra instance (using frobenius norm) for matrices over `ℝ` or `ℂ`.  Not
declared as an instance because there are several natural choices for defining the norm of a
matrix. -/
@[local instance]
def frobeniusNormedAlgebra [DecidableEq m] [NormedField R] [NormedAlgebra R α] :
    NormedAlgebra R (Matrix m m α) :=
  { Matrix.frobeniusNormedSpace, Matrix.instAlgebra with }


