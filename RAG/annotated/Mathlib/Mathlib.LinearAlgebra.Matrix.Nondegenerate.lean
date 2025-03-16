/-- A matrix `M` is nondegenerate if for all `v ≠ 0`, there is a `w ≠ 0` with `w * M * v ≠ 0`. -/
def Nondegenerate (M : Matrix m m R) :=
  ∀ v, (∀ w, dotProduct v (M *ᵥ w) = 0) → v = 0


/-- If `M` is nondegenerate and `w * M * v = 0` for all `w`, then `v = 0`. -/
theorem Nondegenerate.eq_zero_of_ortho {M : Matrix m m R} (hM : Nondegenerate M) {v : m → R}
    (hv : ∀ w, dotProduct v (M *ᵥ w) = 0) : v = 0 :=
  hM v hv


/-- If `M` is nondegenerate and `v ≠ 0`, then there is some `w` such that `w * M * v ≠ 0`. -/
theorem Nondegenerate.exists_not_ortho_of_ne_zero {M : Matrix m m R} (hM : Nondegenerate M)
    {v : m → R} (hv : v ≠ 0) : ∃ w, dotProduct v (M *ᵥ w) ≠ 0 :=
  not_forall.mp (mt hM.eq_zero_of_ortho hv)


/-- If `M` has a nonzero determinant, then `M` as a bilinear form on `n → A` is nondegenerate.

See also `BilinForm.nondegenerateOfDetNeZero'` and `BilinForm.nondegenerateOfDetNeZero`.
-/
theorem nondegenerate_of_det_ne_zero [DecidableEq m] {M : Matrix m m A} (hM : M.det ≠ 0) :
    Nondegenerate M := by
  /-
    m : Type u_1
    A : Type u_3
    inst✝³ : Fintype m
    inst✝² : CommRing A
    inst✝¹ : IsDomain A
    inst✝ : DecidableEq m
    M : Matrix m m A
    hM : Ne M.det 0
    ⊢ M.Nondegenerate
  -/
  intro v hv
  /-
    m : Type u_1
    A : Type u_3
    inst✝³ : Fintype m
    inst✝² : CommRing A
    inst✝¹ : IsDomain A
    inst✝ : DecidableEq m
    M : Matrix m m A
    hM : Ne M.det 0
    v : m → A
    hv : ∀ (w : m → A), Eq (dotProduct v (M.mulVec w)) 0
    ⊢ Eq v 0
  -/
  ext i
  /-
    case h
    m : Type u_1
    A : Type u_3
    inst✝³ : Fintype m
    inst✝² : CommRing A
    inst✝¹ : IsDomain A
    inst✝ : DecidableEq m
    M : Matrix m m A
    hM : Ne M.det 0
    v : m → A
    hv : ∀ (w : m → A), Eq (dotProduct v (M.mulVec w)) 0
    i : m
    ⊢ Eq (v i) (0 i)
  -/
  specialize hv (M.cramer (Pi.single i 1))
  /-
    case h
    m : Type u_1
    A : Type u_3
    inst✝³ : Fintype m
    inst✝² : CommRing A
    inst✝¹ : IsDomain A
    inst✝ : DecidableEq m
    M : Matrix m m A
    hM : Ne M.det 0
    v : m → A
    i : m
    hv : Eq (dotProduct v (M.mulVec (M.cramer (Pi.single i 1)))) 0
    ⊢ Eq (v i) (0 i)
  -/
  refine (mul_eq_zero.mp ?_).resolve_right hM
  /-
    case h
    m : Type u_1
    A : Type u_3
    inst✝³ : Fintype m
    inst✝² : CommRing A
    inst✝¹ : IsDomain A
    inst✝ : DecidableEq m
    M : Matrix m m A
    hM : Ne M.det 0
    v : m → A
    i : m
    hv : Eq (dotProduct v (M.mulVec (M.cramer (Pi.single i 1)))) 0
    ⊢ Eq (HMul.hMul (v i) M.det) 0
  -/
  convert hv
  /-
    case h.e'_2
    m : Type u_1
    A : Type u_3
    inst✝³ : Fintype m
    inst✝² : CommRing A
    inst✝¹ : IsDomain A
    inst✝ : DecidableEq m
    M : Matrix m m A
    hM : Ne M.det 0
    v : m → A
    i : m
    hv : Eq (dotProduct v (M.mulVec (M.cramer (Pi.single i 1)))) 0
    ⊢ Eq (HMul.hMul (v i) M.det) (dotProduct v (M.mulVec (M.cramer (Pi.single i 1) …
  -/
  simp only [mulVec_cramer M (Pi.single i 1), dotProduct, Pi.smul_apply, smul_eq_mul]
  /-
    case h.e'_2
    m : Type u_1
    A : Type u_3
    inst✝³ : Fintype m
    inst✝² : CommRing A
    inst✝¹ : IsDomain A
    inst✝ : DecidableEq m
    M : Matrix m m A
    hM : Ne M.det 0
    v : m → A
    i : m
    hv : Eq (dotProduct v (M.mulVec (M.cramer (Pi.single i 1)))) 0
    ⊢ Eq (HMul.hMul (v i) M.det) (Finset.univ.sum fun x => HMul.hMul (v x) (HMul.h …
  -/
  rw [Finset.sum_eq_single i, Pi.single_eq_same, mul_one]
    /-
      case h.e'_2.h₀
      m : Type u_1
      A : Type u_3
      inst✝³ : Fintype m
      inst✝² : CommRing A
      inst✝¹ : IsDomain A
      inst✝ : DecidableEq m
      M : Matrix m m A
      hM : Ne M.det 0
      v : m → A
      i : m
      hv : Eq (dotProduct v (M.mulVec (M.cramer (Pi.single i 1)))) 0
      ⊢ ∀ (b : m), Membership.mem Finset.univ b → Ne b i → Eq (HMul.hMul (v b) (HMul …
    -/
  · intro j _ hj
    /-
      case h.e'_2.h₀
      m : Type u_1
      A : Type u_3
      inst✝³ : Fintype m
      inst✝² : CommRing A
      inst✝¹ : IsDomain A
      inst✝ : DecidableEq m
      M : Matrix m m A
      hM : Ne M.det 0
      v : m → A
      i : m
      hv : Eq (dotProduct v (M.mulVec (M.cramer (Pi.single i 1)))) 0
      j : m
      a✝ : Membership.mem Finset.univ j
      hj : Ne j i
      ⊢ Eq (HMul.hMul (v j) (HMul.hMul M.det (Pi.single i 1 j))) 0
    -/
    simp [hj]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_2.h₁
      m : Type u_1
      A : Type u_3
      inst✝³ : Fintype m
      inst✝² : CommRing A
      inst✝¹ : IsDomain A
      inst✝ : DecidableEq m
      M : Matrix m m A
      hM : Ne M.det 0
      v : m → A
      i : m
      hv : Eq (dotProduct v (M.mulVec (M.cramer (Pi.single i 1)))) 0
      ⊢ Not (Membership.mem Finset.univ i) → Eq (HMul.hMul (v i) (HMul.hMul M.det (P …
    -/
  · intros
    /-
      case h.e'_2.h₁
      m : Type u_1
      A : Type u_3
      inst✝³ : Fintype m
      inst✝² : CommRing A
      inst✝¹ : IsDomain A
      inst✝ : DecidableEq m
      M : Matrix m m A
      hM : Ne M.det 0
      v : m → A
      i : m
      hv : Eq (dotProduct v (M.mulVec (M.cramer (Pi.single i 1)))) 0
      a✝ : Not (Membership.mem Finset.univ i)
      ⊢ Eq (HMul.hMul (v i) (HMul.hMul M.det (Pi.single i 1 i))) 0
    -/
    have := Finset.mem_univ i
    /-
      case h.e'_2.h₁
      m : Type u_1
      A : Type u_3
      inst✝³ : Fintype m
      inst✝² : CommRing A
      inst✝¹ : IsDomain A
      inst✝ : DecidableEq m
      M : Matrix m m A
      hM : Ne M.det 0
      v : m → A
      i : m
      hv : Eq (dotProduct v (M.mulVec (M.cramer (Pi.single i 1)))) 0
      a✝ : Not (Membership.mem Finset.univ i)
      this : Membership.mem Finset.univ i
      ⊢ Eq (HMul.hMul (v i) (HMul.hMul M.det (Pi.single i 1 i))) 0
    -/
    contradiction
    /-
      🎉 no goals
    -/


theorem eq_zero_of_vecMul_eq_zero [DecidableEq m] {M : Matrix m m A} (hM : M.det ≠ 0) {v : m → A}
    (hv : v ᵥ* M = 0) : v = 0 :=
  (nondegenerate_of_det_ne_zero hM).eq_zero_of_ortho fun w => by
    /-
      m : Type u_1
      A : Type u_3
      inst✝³ : Fintype m
      inst✝² : CommRing A
      inst✝¹ : IsDomain A
      inst✝ : DecidableEq m
      M : Matrix m m A
      hM : Ne M.det 0
      v : m → A
      hv : Eq (Matrix.vecMul v M) 0
      w : m → A
      ⊢ Eq (dotProduct v (M.mulVec w)) 0
    -/
    rw [dotProduct_mulVec, hv, zero_dotProduct]
    /-
      🎉 no goals
    -/


theorem eq_zero_of_mulVec_eq_zero [DecidableEq m] {M : Matrix m m A} (hM : M.det ≠ 0) {v : m → A}
    (hv : M *ᵥ v = 0) : v = 0 :=
                                /-
                                  m : Type u_1
                                  A : Type u_3
                                  inst✝³ : Fintype m
                                  inst✝² : CommRing A
                                  inst✝¹ : IsDomain A
                                  inst✝ : DecidableEq m
                                  M : Matrix m m A
                                  hM : Ne M.det 0
                                  v : m → A
                                  hv : Eq (M.mulVec v) 0
                                  ⊢ Ne M.transpose.det 0
                                -/
  eq_zero_of_vecMul_eq_zero (by rwa [det_transpose]) ((vecMul_transpose M v).trans hv)
                                /-
                                  🎉 no goals
                                -/


