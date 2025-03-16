lemma mem_subfield_of_mul_eq_one_of_mem_subfield_right
    (h_mem : ∀ i j, A i j ∈ K) (i : n) (j : m) :
    B i j ∈ K := by
  /-
    m : Type u_1
    n : Type u_2
    L : Type u_3
    inst✝³ : Fintype m
    inst✝² : Fintype n
    inst✝¹ : DecidableEq m
    inst✝ : Field L
    e : Equiv m n
    K : Subfield L
    A : Matrix m n L
    B : Matrix n m L
    hAB : Eq (HMul.hMul A B) 1
    h_mem : ∀ (i : m) (j : n), Membership.mem K (A i j)
    i : n
    j : m
    ⊢ Membership.mem K (B i j)
  -/
  let A' : Matrix m m K := of fun i j ↦ ⟨A.submatrix id e i j, h_mem i (e j)⟩
  /-
    m : Type u_1
    n : Type u_2
    L : Type u_3
    inst✝³ : Fintype m
    inst✝² : Fintype n
    inst✝¹ : DecidableEq m
    inst✝ : Field L
    e : Equiv m n
    K : Subfield L
    A : Matrix m n L
    B : Matrix n m L
    hAB : Eq (HMul.hMul A B) 1
    h_mem : ∀ (i : m) (j : n), Membership.mem K (A i j)
    i : n
    j : m
    A' : Matrix m m (Subtype fun x => Membership.mem K x) := Matrix.of fun i j =>  …
    ⊢ Membership.mem K (B i j)
  -/
  have hA' : A'.map K.subtype = A.submatrix id e := rfl
  have hA : IsUnit A' := by
    have h_unit : IsUnit (A.submatrix id e) :=
      isUnit_of_right_inverse (B := B.submatrix e id) (by simpa)
    have h_det : (A.submatrix id e).det = K.subtype A'.det := by
      simp [A', K.subtype.map_det, map, submatrix]
    simpa [isUnit_iff_isUnit_det, h_det] using h_unit
  /-
    m : Type u_1
    n : Type u_2
    L : Type u_3
    inst✝³ : Fintype m
    inst✝² : Fintype n
    inst✝¹ : DecidableEq m
    inst✝ : Field L
    e : Equiv m n
    K : Subfield L
    A : Matrix m n L
    B : Matrix n m L
    hAB : Eq (HMul.hMul A B) 1
    h_mem : ∀ (i : m) (j : n), Membership.mem K (A i j)
    i : n
    j : m
    A' : Matrix m m (Subtype fun x => Membership.mem K x) := Matrix.of fun i j =>  …
    hA' : Eq (A'.map ⇑K.subtype) (A.submatrix id ⇑e)
    hA : IsUnit A'
    ⊢ Membership.mem K (B i j)
  -/
  obtain ⟨B', hB⟩ := exists_right_inverse_iff_isUnit.mpr hA
  /-
    case intro
    m : Type u_1
    n : Type u_2
    L : Type u_3
    inst✝³ : Fintype m
    inst✝² : Fintype n
    inst✝¹ : DecidableEq m
    inst✝ : Field L
    e : Equiv m n
    K : Subfield L
    A : Matrix m n L
    B : Matrix n m L
    hAB : Eq (HMul.hMul A B) 1
    h_mem : ∀ (i : m) (j : n), Membership.mem K (A i j)
    i : n
    j : m
    A' : Matrix m m (Subtype fun x => Membership.mem K x) := Matrix.of fun i j =>  …
    hA' : Eq (A'.map ⇑K.subtype) (A.submatrix id ⇑e)
    hA : IsUnit A'
    B' : Matrix m m (Subtype fun x => Membership.mem K x)
    hB : Eq (HMul.hMul A' B') 1
    ⊢ Membership.mem K (B i j)
  -/
  suffices (B'.submatrix e.symm id).map K.subtype = B by simp [← this]
  replace hB : A * (B'.submatrix e.symm id).map K.subtype = 1 := by
    replace hB := congr_arg (fun C ↦ C.map K.subtype) hB
    simp_rw [map_mul] at hB
    rw [hA', ← e.symm_symm, ← submatrix_id_mul_left] at hB
    simpa using hB
  classical
  simpa [← Matrix.mul_assoc, (mul_eq_one_comm_of_equiv e).mp hAB] using congr_arg (B * ·) hB


lemma mem_subfield_of_mul_eq_one_of_mem_subfield_left
    (h_mem : ∀ i j, B i j ∈ K) (i : m) (j : n) :
    A i j ∈ K := by
  /-
    m : Type u_1
    n : Type u_2
    L : Type u_3
    inst✝³ : Fintype m
    inst✝² : Fintype n
    inst✝¹ : DecidableEq m
    inst✝ : Field L
    e : Equiv m n
    K : Subfield L
    A : Matrix m n L
    B : Matrix n m L
    hAB : Eq (HMul.hMul A B) 1
    h_mem : ∀ (i : n) (j : m), Membership.mem K (B i j)
    i : m
    j : n
    ⊢ Membership.mem K (A i j)
  -/
  replace hAB : Bᵀ * Aᵀ = 1 := by simpa using congr_arg transpose hAB
  /-
    m : Type u_1
    n : Type u_2
    L : Type u_3
    inst✝³ : Fintype m
    inst✝² : Fintype n
    inst✝¹ : DecidableEq m
    inst✝ : Field L
    e : Equiv m n
    K : Subfield L
    A : Matrix m n L
    B : Matrix n m L
    h_mem : ∀ (i : n) (j : m), Membership.mem K (B i j)
    i : m
    j : n
    hAB : Eq (HMul.hMul B.transpose A.transpose) 1
    ⊢ Membership.mem K (A i j)
  -/
  rw [← A.transpose_apply]
  /-
    m : Type u_1
    n : Type u_2
    L : Type u_3
    inst✝³ : Fintype m
    inst✝² : Fintype n
    inst✝¹ : DecidableEq m
    inst✝ : Field L
    e : Equiv m n
    K : Subfield L
    A : Matrix m n L
    B : Matrix n m L
    h_mem : ∀ (i : n) (j : m), Membership.mem K (B i j)
    i : m
    j : n
    hAB : Eq (HMul.hMul B.transpose A.transpose) 1
    ⊢ Membership.mem K (A.transpose j i)
  -/
  simp_rw [← B.transpose_apply] at h_mem
  /-
    m : Type u_1
    n : Type u_2
    L : Type u_3
    inst✝³ : Fintype m
    inst✝² : Fintype n
    inst✝¹ : DecidableEq m
    inst✝ : Field L
    e : Equiv m n
    K : Subfield L
    A : Matrix m n L
    B : Matrix n m L
    h_mem : ∀ (i : n) (j : m), Membership.mem K (B.transpose j i)
    i : m
    j : n
    hAB : Eq (HMul.hMul B.transpose A.transpose) 1
    ⊢ Membership.mem K (A.transpose j i)
  -/
  exact mem_subfield_of_mul_eq_one_of_mem_subfield_right e K hAB (fun i j ↦ h_mem j i) j i
  /-
    🎉 no goals
  -/


