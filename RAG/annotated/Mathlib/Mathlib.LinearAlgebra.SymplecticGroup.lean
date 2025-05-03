/-- The matrix defining the canonical skew-symmetric bilinear form. -/
def J : Matrix (l ⊕ l) (l ⊕ l) R :=
  Matrix.fromBlocks 0 (-1) 1 0


@[simp]
theorem J_transpose : (J l R)ᵀ = -J l R := by
  rw [J, fromBlocks_transpose, ← neg_one_smul R (fromBlocks _ _ _ _ : Matrix (l ⊕ l) (l ⊕ l) R),
    fromBlocks_smul, Matrix.transpose_zero, Matrix.transpose_one, transpose_neg]
  /-
    l : Type u_1
    R : Type u_2
    inst✝¹ : DecidableEq l
    inst✝ : CommRing R
    ⊢ Eq (Matrix.fromBlocks 0 1 (Neg.neg (Matrix.transpose 1)) 0) (Matrix.fromBloc …
  -/
  simp [fromBlocks]
  /-
    🎉 no goals
  -/


theorem J_squared : J l R * J l R = -1 := by
  /-
    l : Type u_1
    R : Type u_2
    inst✝² : DecidableEq l
    inst✝¹ : CommRing R
    inst✝ : Fintype l
    ⊢ Eq (HMul.hMul (Matrix.J l R) (Matrix.J l R)) (-1)
  -/
  rw [J, fromBlocks_multiply]
  /-
    l : Type u_1
    R : Type u_2
    inst✝² : DecidableEq l
    inst✝¹ : CommRing R
    inst✝ : Fintype l
    ⊢ Eq (Matrix.fromBlocks (HAdd.hAdd (HMul.hMul 0 0) (HMul.hMul (-1) 1)) (HAdd.h …
  -/
  simp only [Matrix.zero_mul, Matrix.neg_mul, zero_add, neg_zero, Matrix.one_mul, add_zero]
  /-
    l : Type u_1
    R : Type u_2
    inst✝² : DecidableEq l
    inst✝¹ : CommRing R
    inst✝ : Fintype l
    ⊢ Eq (Matrix.fromBlocks (-1) 0 0 (-1)) (-1)
  -/
  rw [← neg_zero, ← Matrix.fromBlocks_neg, ← fromBlocks_one]
  /-
    🎉 no goals
  -/


theorem J_inv : (J l R)⁻¹ = -J l R := by
  /-
    l : Type u_1
    R : Type u_2
    inst✝² : DecidableEq l
    inst✝¹ : CommRing R
    inst✝ : Fintype l
    ⊢ Eq (Inv.inv (Matrix.J l R)) (Neg.neg (Matrix.J l R))
  -/
  refine Matrix.inv_eq_right_inv ?_
  /-
    l : Type u_1
    R : Type u_2
    inst✝² : DecidableEq l
    inst✝¹ : CommRing R
    inst✝ : Fintype l
    ⊢ Eq (HMul.hMul (Matrix.J l R) (Neg.neg (Matrix.J l R))) 1
  -/
  rw [Matrix.mul_neg, J_squared]
  /-
    l : Type u_1
    R : Type u_2
    inst✝² : DecidableEq l
    inst✝¹ : CommRing R
    inst✝ : Fintype l
    ⊢ Eq (Neg.neg (-1)) 1
  -/
  exact neg_neg 1
  /-
    🎉 no goals
  -/


theorem J_det_mul_J_det : det (J l R) * det (J l R) = 1 := by
  rw [← det_mul, J_squared, ← one_smul R (-1 : Matrix _ _ R), smul_neg, ← neg_smul, det_smul,
    Fintype.card_sum, det_one, mul_one]
  /-
    l : Type u_1
    R : Type u_2
    inst✝² : DecidableEq l
    inst✝¹ : CommRing R
    inst✝ : Fintype l
    ⊢ Eq (HPow.hPow (-1) (HAdd.hAdd (Fintype.card l) (Fintype.card l))) 1
  -/
  apply Even.neg_one_pow
  /-
    case h
    l : Type u_1
    R : Type u_2
    inst✝² : DecidableEq l
    inst✝¹ : CommRing R
    inst✝ : Fintype l
    ⊢ Even (HAdd.hAdd (Fintype.card l) (Fintype.card l))
  -/
  exact Even.add_self _
  /-
    🎉 no goals
  -/


theorem isUnit_det_J : IsUnit (det (J l R)) :=
  isUnit_iff_exists_inv.mpr ⟨det (J l R), J_det_mul_J_det _ _⟩


/-- The group of symplectic matrices over a ring `R`. -/
def symplecticGroup : Submonoid (Matrix (l ⊕ l) (l ⊕ l) R) where
  carrier := { A | A * J l R * Aᵀ = J l R }
  mul_mem' {a b} ha hb := by
    /-
      l : Type u_1
      R : Type u_2
      inst✝² : DecidableEq l
      inst✝¹ : CommRing R
      inst✝ : Fintype l
      a b : Matrix (Sum l l) (Sum l l) R
      ha : Membership.mem (setOf fun A => Eq (HMul.hMul (HMul.hMul A (Matrix.J l R)) …
      hb : Membership.mem (setOf fun A => Eq (HMul.hMul (HMul.hMul A (Matrix.J l R)) …
      ⊢ Membership.mem (setOf fun A => Eq (HMul.hMul (HMul.hMul A (Matrix.J l R)) A. …
    -/
    simp only [Set.mem_setOf_eq, transpose_mul] at *
    /-
      l : Type u_1
      R : Type u_2
      inst✝² : DecidableEq l
      inst✝¹ : CommRing R
      inst✝ : Fintype l
      a b : Matrix (Sum l l) (Sum l l) R
      ha : Eq (HMul.hMul (HMul.hMul a (Matrix.J l R)) a.transpose) (Matrix.J l R)
      hb : Eq (HMul.hMul (HMul.hMul b (Matrix.J l R)) b.transpose) (Matrix.J l R)
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul a b) (Matrix.J l R)) (HMul.hMul b.transp …
    -/
    rw [← Matrix.mul_assoc, a.mul_assoc, a.mul_assoc, hb]
    /-
      l : Type u_1
      R : Type u_2
      inst✝² : DecidableEq l
      inst✝¹ : CommRing R
      inst✝ : Fintype l
      a b : Matrix (Sum l l) (Sum l l) R
      ha : Eq (HMul.hMul (HMul.hMul a (Matrix.J l R)) a.transpose) (Matrix.J l R)
      hb : Eq (HMul.hMul (HMul.hMul b (Matrix.J l R)) b.transpose) (Matrix.J l R)
      ⊢ Eq (HMul.hMul (HMul.hMul a (Matrix.J l R)) a.transpose) (Matrix.J l R)
    -/
    exact ha
    /-
      🎉 no goals
    -/
                 /-
                   l : Type u_1
                   R : Type u_2
                   inst✝² : DecidableEq l
                   inst✝¹ : CommRing R
                   inst✝ : Fintype l
                   ⊢ Membership.mem { carrier := setOf fun A => Eq (HMul.hMul (HMul.hMul A (Matri …
                 -/
  one_mem' := by simp
                 /-
                   🎉 no goals
                 -/


theorem mem_iff {A : Matrix (l ⊕ l) (l ⊕ l) R} :
                                                           /-
                                                             l : Type u_1
                                                             R : Type u_2
                                                             inst✝² : DecidableEq l
                                                             inst✝¹ : Fintype l
                                                             inst✝ : CommRing R
                                                             A : Matrix (Sum l l) (Sum l l) R
                                                             ⊢ Iff (Membership.mem (Matrix.symplecticGroup l R) A) (Eq (HMul.hMul (HMul.hMu …
                                                           -/
    A ∈ symplecticGroup l R ↔ A * J l R * Aᵀ = J l R := by simp [symplecticGroup]
                                                           /-
                                                             🎉 no goals
                                                           -/


instance coeMatrix : Coe (symplecticGroup l R) (Matrix (l ⊕ l) (l ⊕ l) R) :=
  ⟨Subtype.val⟩


theorem J_mem : J l R ∈ symplecticGroup l R := by
  /-
    l : Type u_1
    R : Type u_2
    inst✝² : DecidableEq l
    inst✝¹ : Fintype l
    inst✝ : CommRing R
    ⊢ Membership.mem (Matrix.symplecticGroup l R) (Matrix.J l R)
  -/
  rw [mem_iff, J, fromBlocks_multiply, fromBlocks_transpose, fromBlocks_multiply]
  /-
    l : Type u_1
    R : Type u_2
    inst✝² : DecidableEq l
    inst✝¹ : Fintype l
    inst✝ : CommRing R
    ⊢ Eq (Matrix.fromBlocks (HAdd.hAdd (HMul.hMul (HAdd.hAdd (HMul.hMul 0 0) (HMul …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The canonical skew-symmetric matrix as an element in the symplectic group. -/
def symJ : symplecticGroup l R :=
  ⟨J l R, J_mem l R⟩


@[simp]
theorem coe_J : ↑(symJ l R) = J l R := rfl


theorem neg_mem (h : A ∈ symplecticGroup l R) : -A ∈ symplecticGroup l R := by
  /-
    l : Type u_1
    R : Type u_2
    inst✝² : DecidableEq l
    inst✝¹ : Fintype l
    inst✝ : CommRing R
    A : Matrix (Sum l l) (Sum l l) R
    h : Membership.mem (Matrix.symplecticGroup l R) A
    ⊢ Membership.mem (Matrix.symplecticGroup l R) (Neg.neg A)
  -/
  rw [mem_iff] at h ⊢
  /-
    l : Type u_1
    R : Type u_2
    inst✝² : DecidableEq l
    inst✝¹ : Fintype l
    inst✝ : CommRing R
    A : Matrix (Sum l l) (Sum l l) R
    h : Eq (HMul.hMul (HMul.hMul A (Matrix.J l R)) A.transpose) (Matrix.J l R)
    ⊢ Eq (HMul.hMul (HMul.hMul (Neg.neg A) (Matrix.J l R)) (Neg.neg A).transpose)  …
  -/
  simp [h]
  /-
    🎉 no goals
  -/


theorem symplectic_det (hA : A ∈ symplecticGroup l R) : IsUnit <| det A := by
  /-
    l : Type u_1
    R : Type u_2
    inst✝² : DecidableEq l
    inst✝¹ : Fintype l
    inst✝ : CommRing R
    A : Matrix (Sum l l) (Sum l l) R
    hA : Membership.mem (Matrix.symplecticGroup l R) A
    ⊢ IsUnit A.det
  -/
  rw [isUnit_iff_exists_inv]
  /-
    l : Type u_1
    R : Type u_2
    inst✝² : DecidableEq l
    inst✝¹ : Fintype l
    inst✝ : CommRing R
    A : Matrix (Sum l l) (Sum l l) R
    hA : Membership.mem (Matrix.symplecticGroup l R) A
    ⊢ Exists fun b => Eq (HMul.hMul A.det b) 1
  -/
  use A.det
  /-
    case h
    l : Type u_1
    R : Type u_2
    inst✝² : DecidableEq l
    inst✝¹ : Fintype l
    inst✝ : CommRing R
    A : Matrix (Sum l l) (Sum l l) R
    hA : Membership.mem (Matrix.symplecticGroup l R) A
    ⊢ Eq (HMul.hMul A.det A.det) 1
  -/
  refine (isUnit_det_J l R).mul_left_cancel ?_
  /-
    case h
    l : Type u_1
    R : Type u_2
    inst✝² : DecidableEq l
    inst✝¹ : Fintype l
    inst✝ : CommRing R
    A : Matrix (Sum l l) (Sum l l) R
    hA : Membership.mem (Matrix.symplecticGroup l R) A
    ⊢ Eq (HMul.hMul (Matrix.J l R).det (HMul.hMul A.det A.det)) (HMul.hMul (Matrix …
  -/
  rw [mul_one]
  /-
    case h
    l : Type u_1
    R : Type u_2
    inst✝² : DecidableEq l
    inst✝¹ : Fintype l
    inst✝ : CommRing R
    A : Matrix (Sum l l) (Sum l l) R
    hA : Membership.mem (Matrix.symplecticGroup l R) A
    ⊢ Eq (HMul.hMul (Matrix.J l R).det (HMul.hMul A.det A.det)) (Matrix.J l R).det
  -/
  rw [mem_iff] at hA
  /-
    case h
    l : Type u_1
    R : Type u_2
    inst✝² : DecidableEq l
    inst✝¹ : Fintype l
    inst✝ : CommRing R
    A : Matrix (Sum l l) (Sum l l) R
    hA : Eq (HMul.hMul (HMul.hMul A (Matrix.J l R)) A.transpose) (Matrix.J l R)
    ⊢ Eq (HMul.hMul (Matrix.J l R).det (HMul.hMul A.det A.det)) (Matrix.J l R).det
  -/
  apply_fun det at hA
  /-
    case h
    l : Type u_1
    R : Type u_2
    inst✝² : DecidableEq l
    inst✝¹ : Fintype l
    inst✝ : CommRing R
    A : Matrix (Sum l l) (Sum l l) R
    hA : Eq (HMul.hMul (HMul.hMul A (Matrix.J l R)) A.transpose).det (Matrix.J l R …
    ⊢ Eq (HMul.hMul (Matrix.J l R).det (HMul.hMul A.det A.det)) (Matrix.J l R).det
  -/
  simp only [det_mul, det_transpose] at hA
  /-
    case h
    l : Type u_1
    R : Type u_2
    inst✝² : DecidableEq l
    inst✝¹ : Fintype l
    inst✝ : CommRing R
    A : Matrix (Sum l l) (Sum l l) R
    hA : Eq (HMul.hMul (HMul.hMul A.det (Matrix.J l R).det) A.det) (Matrix.J l R). …
    ⊢ Eq (HMul.hMul (Matrix.J l R).det (HMul.hMul A.det A.det)) (Matrix.J l R).det
  -/
  rw [mul_comm A.det, mul_assoc] at hA
  /-
    case h
    l : Type u_1
    R : Type u_2
    inst✝² : DecidableEq l
    inst✝¹ : Fintype l
    inst✝ : CommRing R
    A : Matrix (Sum l l) (Sum l l) R
    hA : Eq (HMul.hMul (Matrix.J l R).det (HMul.hMul A.det A.det)) (Matrix.J l R). …
    ⊢ Eq (HMul.hMul (Matrix.J l R).det (HMul.hMul A.det A.det)) (Matrix.J l R).det
  -/
  exact hA
  /-
    🎉 no goals
  -/


theorem transpose_mem (hA : A ∈ symplecticGroup l R) : Aᵀ ∈ symplecticGroup l R := by
  /-
    l : Type u_1
    R : Type u_2
    inst✝² : DecidableEq l
    inst✝¹ : Fintype l
    inst✝ : CommRing R
    A : Matrix (Sum l l) (Sum l l) R
    hA : Membership.mem (Matrix.symplecticGroup l R) A
    ⊢ Membership.mem (Matrix.symplecticGroup l R) A.transpose
  -/
  rw [mem_iff] at hA ⊢
  /-
    l : Type u_1
    R : Type u_2
    inst✝² : DecidableEq l
    inst✝¹ : Fintype l
    inst✝ : CommRing R
    A : Matrix (Sum l l) (Sum l l) R
    hA : Eq (HMul.hMul (HMul.hMul A (Matrix.J l R)) A.transpose) (Matrix.J l R)
    ⊢ Eq (HMul.hMul (HMul.hMul A.transpose (Matrix.J l R)) A.transpose.transpose)  …
  -/
  rw [transpose_transpose]
  /-
    l : Type u_1
    R : Type u_2
    inst✝² : DecidableEq l
    inst✝¹ : Fintype l
    inst✝ : CommRing R
    A : Matrix (Sum l l) (Sum l l) R
    hA : Eq (HMul.hMul (HMul.hMul A (Matrix.J l R)) A.transpose) (Matrix.J l R)
    ⊢ Eq (HMul.hMul (HMul.hMul A.transpose (Matrix.J l R)) A) (Matrix.J l R)
  -/
  have huA := symplectic_det hA
  have huAT : IsUnit Aᵀ.det := by
    rw [Matrix.det_transpose]
    exact huA
  calc
    Aᵀ * J l R * A = (-Aᵀ) * (J l R)⁻¹ * A := by
      rw [J_inv]
      simp
    _ = (-Aᵀ) * (A * J l R * Aᵀ)⁻¹ * A := by rw [hA]
    _ = -(Aᵀ * (Aᵀ⁻¹ * (J l R)⁻¹)) * A⁻¹ * A := by
      simp only [Matrix.mul_inv_rev, Matrix.mul_assoc, Matrix.neg_mul]
    _ = -(J l R)⁻¹ := by
      rw [mul_nonsing_inv_cancel_left _ _ huAT, nonsing_inv_mul_cancel_right _ _ huA]
    _ = J l R := by simp [J_inv]


@[simp]
theorem transpose_mem_iff : Aᵀ ∈ symplecticGroup l R ↔ A ∈ symplecticGroup l R :=
                /-
                  l : Type u_1
                  R : Type u_2
                  inst✝² : DecidableEq l
                  inst✝¹ : Fintype l
                  inst✝ : CommRing R
                  A : Matrix (Sum l l) (Sum l l) R
                  hA : Membership.mem (Matrix.symplecticGroup l R) A.transpose
                  ⊢ Membership.mem (Matrix.symplecticGroup l R) A
                -/
  ⟨fun hA => by simpa using transpose_mem hA, transpose_mem⟩
                /-
                  🎉 no goals
                -/


theorem mem_iff' : A ∈ symplecticGroup l R ↔ Aᵀ * J l R * A = J l R := by
  /-
    l : Type u_1
    R : Type u_2
    inst✝² : DecidableEq l
    inst✝¹ : Fintype l
    inst✝ : CommRing R
    A : Matrix (Sum l l) (Sum l l) R
    ⊢ Iff (Membership.mem (Matrix.symplecticGroup l R) A) (Eq (HMul.hMul (HMul.hMu …
  -/
  rw [← transpose_mem_iff, mem_iff, transpose_transpose]
  /-
    🎉 no goals
  -/


instance hasInv : Inv (symplecticGroup l R) where
  inv A := ⟨(-J l R) * (A : Matrix (l ⊕ l) (l ⊕ l) R)ᵀ * J l R,
      mul_mem (mul_mem (neg_mem <| J_mem _ _) <| transpose_mem A.2) <| J_mem _ _⟩


theorem coe_inv (A : symplecticGroup l R) : (↑A⁻¹ : Matrix _ _ _) = (-J l R) * (↑A)ᵀ * J l R := rfl


theorem inv_left_mul_aux (hA : A ∈ symplecticGroup l R) : -(J l R * Aᵀ * J l R * A) = 1 :=
  calc
    -(J l R * Aᵀ * J l R * A) = (-J l R) * (Aᵀ * J l R * A) := by
      /-
        l : Type u_1
        R : Type u_2
        inst✝² : DecidableEq l
        inst✝¹ : Fintype l
        inst✝ : CommRing R
        A : Matrix (Sum l l) (Sum l l) R
        hA : Membership.mem (Matrix.symplecticGroup l R) A
        ⊢ Eq (Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul (Matrix.J l R) A.transpose) (Ma …
      -/
      simp only [Matrix.mul_assoc, Matrix.neg_mul]
      /-
        🎉 no goals
      -/
    _ = (-J l R) * J l R := by
      /-
        l : Type u_1
        R : Type u_2
        inst✝² : DecidableEq l
        inst✝¹ : Fintype l
        inst✝ : CommRing R
        A : Matrix (Sum l l) (Sum l l) R
        hA : Membership.mem (Matrix.symplecticGroup l R) A
        ⊢ Eq (HMul.hMul (Neg.neg (Matrix.J l R)) (HMul.hMul (HMul.hMul A.transpose (Ma …
      -/
      rw [mem_iff'] at hA
      /-
        l : Type u_1
        R : Type u_2
        inst✝² : DecidableEq l
        inst✝¹ : Fintype l
        inst✝ : CommRing R
        A : Matrix (Sum l l) (Sum l l) R
        hA : Eq (HMul.hMul (HMul.hMul A.transpose (Matrix.J l R)) A) (Matrix.J l R)
        ⊢ Eq (HMul.hMul (Neg.neg (Matrix.J l R)) (HMul.hMul (HMul.hMul A.transpose (Ma …
      -/
      rw [hA]
      /-
        🎉 no goals
      -/
                                         /-
                                           l : Type u_1
                                           R : Type u_2
                                           inst✝² : DecidableEq l
                                           inst✝¹ : Fintype l
                                           inst✝ : CommRing R
                                           A : Matrix (Sum l l) (Sum l l) R
                                           hA : Membership.mem (Matrix.symplecticGroup l R) A
                                           ⊢ Eq (HMul.hMul (Neg.neg (Matrix.J l R)) (Matrix.J l R)) (HSMul.hSMul (-1) (HM …
                                         -/
    _ = (-1 : R) • (J l R * J l R) := by simp only [Matrix.neg_mul, neg_smul, one_smul]
                                         /-
                                           🎉 no goals
                                         -/
                                             /-
                                               l : Type u_1
                                               R : Type u_2
                                               inst✝² : DecidableEq l
                                               inst✝¹ : Fintype l
                                               inst✝ : CommRing R
                                               A : Matrix (Sum l l) (Sum l l) R
                                               hA : Membership.mem (Matrix.symplecticGroup l R) A
                                               ⊢ Eq (HSMul.hSMul (-1) (HMul.hMul (Matrix.J l R) (Matrix.J l R))) (HSMul.hSMul …
                                             -/
    _ = (-1 : R) • (-1 : Matrix _ _ _) := by rw [J_squared]
                                             /-
                                               🎉 no goals
                                             -/
                /-
                  l : Type u_1
                  R : Type u_2
                  inst✝² : DecidableEq l
                  inst✝¹ : Fintype l
                  inst✝ : CommRing R
                  A : Matrix (Sum l l) (Sum l l) R
                  hA : Membership.mem (Matrix.symplecticGroup l R) A
                  ⊢ Eq (HSMul.hSMul (-1) (-1)) 1
                -/
    _ = 1 := by simp only [neg_smul_neg, one_smul]
                /-
                  🎉 no goals
                -/


theorem coe_inv' (A : symplecticGroup l R) : (↑A⁻¹ : Matrix (l ⊕ l) (l ⊕ l) R) = (↑A)⁻¹ := by
  /-
    l : Type u_1
    R : Type u_2
    inst✝² : DecidableEq l
    inst✝¹ : Fintype l
    inst✝ : CommRing R
    A : Subtype fun x => Membership.mem (Matrix.symplecticGroup l R) x
    ⊢ Eq (↑(Inv.inv A)) (Inv.inv ↑A)
  -/
  refine (coe_inv A).trans (inv_eq_left_inv ?_).symm
  /-
    l : Type u_1
    R : Type u_2
    inst✝² : DecidableEq l
    inst✝¹ : Fintype l
    inst✝ : CommRing R
    A : Subtype fun x => Membership.mem (Matrix.symplecticGroup l R) x
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (Neg.neg (Matrix.J l R)) (↑A).transpose) …
  -/
  simp [inv_left_mul_aux, coe_inv]
  /-
    🎉 no goals
  -/


theorem inv_eq_symplectic_inv (A : Matrix (l ⊕ l) (l ⊕ l) R) (hA : A ∈ symplecticGroup l R) :
    A⁻¹ = (-J l R) * Aᵀ * J l R :=
                      /-
                        l : Type u_1
                        R : Type u_2
                        inst✝² : DecidableEq l
                        inst✝¹ : Fintype l
                        inst✝ : CommRing R
                        A : Matrix (Sum l l) (Sum l l) R
                        hA : Membership.mem (Matrix.symplecticGroup l R) A
                        ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (Neg.neg (Matrix.J l R)) A.transpose) (M …
                      -/
  inv_eq_left_inv (by simp only [Matrix.neg_mul, inv_left_mul_aux hA])
                      /-
                        🎉 no goals
                      -/


instance : Group (symplecticGroup l R) :=
  { SymplecticGroup.hasInv, Submonoid.toMonoid _ with
    inv_mul_cancel := fun A => by
      /-
        l : Type u_1
        R : Type u_2
        inst✝² : DecidableEq l
        inst✝¹ : Fintype l
        inst✝ : CommRing R
        A✝ : Matrix (Sum l l) (Sum l l) R
        A : Subtype fun x => Membership.mem (Matrix.symplecticGroup l R) x
        ⊢ Eq (HMul.hMul (Inv.inv A) A) 1
      -/
      apply Subtype.ext
      /-
        case a
        l : Type u_1
        R : Type u_2
        inst✝² : DecidableEq l
        inst✝¹ : Fintype l
        inst✝ : CommRing R
        A✝ : Matrix (Sum l l) (Sum l l) R
        A : Subtype fun x => Membership.mem (Matrix.symplecticGroup l R) x
        ⊢ Eq ↑(HMul.hMul (Inv.inv A) A) ↑1
      -/
      simp only [Submonoid.coe_one, Submonoid.coe_mul, Matrix.neg_mul, coe_inv]
      /-
        case a
        l : Type u_1
        R : Type u_2
        inst✝² : DecidableEq l
        inst✝¹ : Fintype l
        inst✝ : CommRing R
        A✝ : Matrix (Sum l l) (Sum l l) R
        A : Subtype fun x => Membership.mem (Matrix.symplecticGroup l R) x
        ⊢ Eq (Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul (Matrix.J l R) (↑A).transpose)  …
      -/
      exact inv_left_mul_aux A.2 }
      /-
        🎉 no goals
      -/


