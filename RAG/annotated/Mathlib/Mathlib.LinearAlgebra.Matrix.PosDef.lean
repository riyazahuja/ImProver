/-- A matrix `M : Matrix n n R` is positive semidefinite if it is Hermitian and `xᴴ * M * x` is
nonnegative for all `x`. -/
def PosSemidef (M : Matrix n n R) :=
  M.IsHermitian ∧ ∀ x : n → R, 0 ≤ dotProduct (star x) (M *ᵥ x)


protected theorem PosSemidef.diagonal [StarOrderedRing R] [DecidableEq n] {d : n → R} (h : 0 ≤ d) :
    PosSemidef (diagonal d) :=
  ⟨isHermitian_diagonal_of_self_adjoint _ <| funext fun i => IsSelfAdjoint.of_nonneg (h i),
    fun x => by
      /-
        n : Type u_2
        R : Type u_3
        inst✝⁵ : Fintype n
        inst✝⁴ : CommRing R
        inst✝³ : PartialOrder R
        inst✝² : StarRing R
        inst✝¹ : StarOrderedRing R
        inst✝ : DecidableEq n
        d : n → R
        h : LE.le 0 d
        x : n → R
        ⊢ LE.le 0 (dotProduct (Star.star x) ((Matrix.diagonal d).mulVec x))
      -/
      refine Fintype.sum_nonneg fun i => ?_
      /-
        n : Type u_2
        R : Type u_3
        inst✝⁵ : Fintype n
        inst✝⁴ : CommRing R
        inst✝³ : PartialOrder R
        inst✝² : StarRing R
        inst✝¹ : StarOrderedRing R
        inst✝ : DecidableEq n
        d : n → R
        h : LE.le 0 d
        x : n → R
        i : n
        ⊢ LE.le (0 i) (HMul.hMul (Star.star x i) ((Matrix.diagonal d).mulVec x i))
      -/
      simpa only [mulVec_diagonal, ← mul_assoc] using conjugate_nonneg (h i) _⟩
      /-
        🎉 no goals
      -/


/-- A diagonal matrix is positive semidefinite iff its diagonal entries are nonnegative. -/
lemma posSemidef_diagonal_iff [StarOrderedRing R] [DecidableEq n] {d : n → R} :
    PosSemidef (diagonal d) ↔ (∀ i : n, 0 ≤ d i) :=
                      /-
                        n : Type u_2
                        R : Type u_3
                        inst✝⁵ : Fintype n
                        inst✝⁴ : CommRing R
                        inst✝³ : PartialOrder R
                        inst✝² : StarRing R
                        inst✝¹ : StarOrderedRing R
                        inst✝ : DecidableEq n
                        d : n → R
                        x✝ : (Matrix.diagonal d).PosSemidef
                        i : n
                        left✝ : (Matrix.diagonal d).IsHermitian
                        hP : ∀ (x : n → R), LE.le 0 (dotProduct (Star.star x) ((Matrix.diagonal d).mul …
                        ⊢ LE.le 0 (d i)
                      -/
  ⟨fun ⟨_, hP⟩ i ↦ by simpa using hP (Pi.single i 1), .diagonal⟩
                      /-
                        🎉 no goals
                      -/


theorem isHermitian {M : Matrix n n R} (hM : M.PosSemidef) : M.IsHermitian :=
  hM.1


theorem re_dotProduct_nonneg {M : Matrix n n 𝕜} (hM : M.PosSemidef) (x : n → 𝕜) :
    0 ≤ RCLike.re (dotProduct (star x) (M *ᵥ x)) :=
  RCLike.nonneg_iff.mp (hM.2 _) |>.1


lemma conjTranspose_mul_mul_same {A : Matrix n n R} (hA : PosSemidef A)
    {m : Type*} [Fintype m] (B : Matrix n m R) :
    PosSemidef (Bᴴ * A * B) := by
  /-
    n : Type u_2
    R : Type u_3
    inst✝⁴ : Fintype n
    inst✝³ : CommRing R
    inst✝² : PartialOrder R
    inst✝¹ : StarRing R
    A : Matrix n n R
    hA : A.PosSemidef
    m : Type u_5
    inst✝ : Fintype m
    B : Matrix n m R
    ⊢ (HMul.hMul (HMul.hMul B.conjTranspose A) B).PosSemidef
  -/
  constructor
    /-
      case left
      n : Type u_2
      R : Type u_3
      inst✝⁴ : Fintype n
      inst✝³ : CommRing R
      inst✝² : PartialOrder R
      inst✝¹ : StarRing R
      A : Matrix n n R
      hA : A.PosSemidef
      m : Type u_5
      inst✝ : Fintype m
      B : Matrix n m R
      ⊢ (HMul.hMul (HMul.hMul B.conjTranspose A) B).IsHermitian
    -/
  · exact isHermitian_conjTranspose_mul_mul B hA.1
    /-
      🎉 no goals
    -/
    /-
      case right
      n : Type u_2
      R : Type u_3
      inst✝⁴ : Fintype n
      inst✝³ : CommRing R
      inst✝² : PartialOrder R
      inst✝¹ : StarRing R
      A : Matrix n n R
      hA : A.PosSemidef
      m : Type u_5
      inst✝ : Fintype m
      B : Matrix n m R
      ⊢ ∀ (x : m → R), LE.le 0 (dotProduct (Star.star x) ((HMul.hMul (HMul.hMul B.co …
    -/
  · intro x
    /-
      case right
      n : Type u_2
      R : Type u_3
      inst✝⁴ : Fintype n
      inst✝³ : CommRing R
      inst✝² : PartialOrder R
      inst✝¹ : StarRing R
      A : Matrix n n R
      hA : A.PosSemidef
      m : Type u_5
      inst✝ : Fintype m
      B : Matrix n m R
      x : m → R
      ⊢ LE.le 0 (dotProduct (Star.star x) ((HMul.hMul (HMul.hMul B.conjTranspose A)  …
    -/
    simpa only [star_mulVec, dotProduct_mulVec, vecMul_vecMul] using hA.2 (B *ᵥ x)
    /-
      🎉 no goals
    -/


lemma mul_mul_conjTranspose_same {A : Matrix n n R} (hA : PosSemidef A)
    {m : Type*} [Fintype m] (B : Matrix m n R) :
    PosSemidef (B * A * Bᴴ) := by
  /-
    n : Type u_2
    R : Type u_3
    inst✝⁴ : Fintype n
    inst✝³ : CommRing R
    inst✝² : PartialOrder R
    inst✝¹ : StarRing R
    A : Matrix n n R
    hA : A.PosSemidef
    m : Type u_5
    inst✝ : Fintype m
    B : Matrix m n R
    ⊢ (HMul.hMul (HMul.hMul B A) B.conjTranspose).PosSemidef
  -/
  simpa only [conjTranspose_conjTranspose] using hA.conjTranspose_mul_mul_same Bᴴ
  /-
    🎉 no goals
  -/


theorem submatrix {M : Matrix n n R} (hM : M.PosSemidef) (e : m → n) :
    (M.submatrix e e).PosSemidef := by
  classical
  rw [(by simp : M = 1 * M * 1), submatrix_mul (he₂ := Function.bijective_id),
    submatrix_mul (he₂ := Function.bijective_id), submatrix_id_id]
  simpa only [conjTranspose_submatrix, conjTranspose_one] using
    conjTranspose_mul_mul_same hM (Matrix.submatrix 1 id e)


theorem transpose {M : Matrix n n R} (hM : M.PosSemidef) : Mᵀ.PosSemidef := by
  /-
    n : Type u_2
    R : Type u_3
    inst✝³ : Fintype n
    inst✝² : CommRing R
    inst✝¹ : PartialOrder R
    inst✝ : StarRing R
    M : Matrix n n R
    hM : M.PosSemidef
    ⊢ M.transpose.PosSemidef
  -/
  refine ⟨IsHermitian.transpose hM.1, fun x => ?_⟩
  /-
    n : Type u_2
    R : Type u_3
    inst✝³ : Fintype n
    inst✝² : CommRing R
    inst✝¹ : PartialOrder R
    inst✝ : StarRing R
    M : Matrix n n R
    hM : M.PosSemidef
    x : n → R
    ⊢ LE.le 0 (dotProduct (Star.star x) (M.transpose.mulVec x))
  -/
  convert hM.2 (star x) using 1
  /-
    case h.e'_4
    n : Type u_2
    R : Type u_3
    inst✝³ : Fintype n
    inst✝² : CommRing R
    inst✝¹ : PartialOrder R
    inst✝ : StarRing R
    M : Matrix n n R
    hM : M.PosSemidef
    x : n → R
    ⊢ Eq (dotProduct (Star.star x) (M.transpose.mulVec x)) (dotProduct (Star.star  …
  -/
  rw [mulVec_transpose, dotProduct_mulVec, star_star, dotProduct_comm]
  /-
    🎉 no goals
  -/


theorem conjTranspose {M : Matrix n n R} (hM : M.PosSemidef) : Mᴴ.PosSemidef := hM.1.symm ▸ hM


protected lemma zero : PosSemidef (0 : Matrix n n R) :=
                        /-
                          n : Type u_2
                          R : Type u_3
                          inst✝³ : Fintype n
                          inst✝² : CommRing R
                          inst✝¹ : PartialOrder R
                          inst✝ : StarRing R
                          ⊢ ∀ (x : n → R), LE.le 0 (dotProduct (Star.star x) (Matrix.mulVec 0 x))
                        -/
  ⟨isHermitian_zero, by simp⟩
                        /-
                          🎉 no goals
                        -/


protected lemma one [StarOrderedRing R] [DecidableEq n] : PosSemidef (1 : Matrix n n R) :=
  ⟨isHermitian_one, fun x => by
    /-
      n : Type u_2
      R : Type u_3
      inst✝⁵ : Fintype n
      inst✝⁴ : CommRing R
      inst✝³ : PartialOrder R
      inst✝² : StarRing R
      inst✝¹ : StarOrderedRing R
      inst✝ : DecidableEq n
      x : n → R
      ⊢ LE.le 0 (dotProduct (Star.star x) (Matrix.mulVec 1 x))
    -/
    rw [one_mulVec]; exact Fintype.sum_nonneg fun i => star_mul_self_nonneg _⟩
                     /-
                       🎉 no goals
                     -/


protected theorem natCast [StarOrderedRing R] [DecidableEq n] (d : ℕ) :
    PosSemidef (d : Matrix n n R) :=
  ⟨isHermitian_natCast _, fun x => by
    /-
      n : Type u_2
      R : Type u_3
      inst✝⁵ : Fintype n
      inst✝⁴ : CommRing R
      inst✝³ : PartialOrder R
      inst✝² : StarRing R
      inst✝¹ : StarOrderedRing R
      inst✝ : DecidableEq n
      d : Nat
      x : n → R
      ⊢ LE.le 0 (dotProduct (Star.star x) ((↑d).mulVec x))
    -/
    simp only [natCast_mulVec, dotProduct_smul]
    /-
      n : Type u_2
      R : Type u_3
      inst✝⁵ : Fintype n
      inst✝⁴ : CommRing R
      inst✝³ : PartialOrder R
      inst✝² : StarRing R
      inst✝¹ : StarOrderedRing R
      inst✝ : DecidableEq n
      d : Nat
      x : n → R
      ⊢ LE.le 0 (HSMul.hSMul (↑d) (dotProduct (Star.star x) x))
    -/
    rw [Nat.cast_smul_eq_nsmul]
    /-
      n : Type u_2
      R : Type u_3
      inst✝⁵ : Fintype n
      inst✝⁴ : CommRing R
      inst✝³ : PartialOrder R
      inst✝² : StarRing R
      inst✝¹ : StarOrderedRing R
      inst✝ : DecidableEq n
      d : Nat
      x : n → R
      ⊢ LE.le 0 (HSMul.hSMul d (dotProduct (Star.star x) x))
    -/
    exact nsmul_nonneg (dotProduct_star_self_nonneg _) _⟩
    /-
      🎉 no goals
    -/

-- See note [no_index around OfNat.ofNat]

protected theorem ofNat [StarOrderedRing R] [DecidableEq n] (d : ℕ) [d.AtLeastTwo] :
    PosSemidef (no_index (OfNat.ofNat d) : Matrix n n R) :=
  .natCast d


protected theorem intCast [StarOrderedRing R] [DecidableEq n] (d : ℤ) (hd : 0 ≤ d) :
    PosSemidef (d : Matrix n n R) :=
  ⟨isHermitian_intCast _, fun x => by
    /-
      n : Type u_2
      R : Type u_3
      inst✝⁵ : Fintype n
      inst✝⁴ : CommRing R
      inst✝³ : PartialOrder R
      inst✝² : StarRing R
      inst✝¹ : StarOrderedRing R
      inst✝ : DecidableEq n
      d : Int
      hd : LE.le 0 d
      x : n → R
      ⊢ LE.le 0 (dotProduct (Star.star x) ((↑d).mulVec x))
    -/
    simp only [intCast_mulVec, dotProduct_smul]
    /-
      n : Type u_2
      R : Type u_3
      inst✝⁵ : Fintype n
      inst✝⁴ : CommRing R
      inst✝³ : PartialOrder R
      inst✝² : StarRing R
      inst✝¹ : StarOrderedRing R
      inst✝ : DecidableEq n
      d : Int
      hd : LE.le 0 d
      x : n → R
      ⊢ LE.le 0 (HSMul.hSMul (↑d) (dotProduct (Star.star x) x))
    -/
    rw [Int.cast_smul_eq_zsmul]
    /-
      n : Type u_2
      R : Type u_3
      inst✝⁵ : Fintype n
      inst✝⁴ : CommRing R
      inst✝³ : PartialOrder R
      inst✝² : StarRing R
      inst✝¹ : StarOrderedRing R
      inst✝ : DecidableEq n
      d : Int
      hd : LE.le 0 d
      x : n → R
      ⊢ LE.le 0 (HSMul.hSMul d (dotProduct (Star.star x) x))
    -/
    exact zsmul_nonneg (dotProduct_star_self_nonneg _) hd⟩
    /-
      🎉 no goals
    -/


@[simp]
protected theorem _root_.Matrix.posSemidef_intCast_iff
    [StarOrderedRing R] [DecidableEq n] [Nonempty n] [Nontrivial R] (d : ℤ) :
    PosSemidef (d : Matrix n n R) ↔ 0 ≤ d :=
                                      /-
                                        n : Type u_2
                                        R : Type u_3
                                        inst✝⁷ : Fintype n
                                        inst✝⁶ : CommRing R
                                        inst✝⁵ : PartialOrder R
                                        inst✝⁴ : StarRing R
                                        inst✝³ : StarOrderedRing R
                                        inst✝² : DecidableEq n
                                        inst✝¹ : Nonempty n
                                        inst✝ : Nontrivial R
                                        d : Int
                                        ⊢ Iff (n → LE.le 0 ↑d) (LE.le 0 d)
                                      -/
  posSemidef_diagonal_iff.trans <| by simp [Pi.le_def]
                                      /-
                                        🎉 no goals
                                      -/


protected lemma pow [StarOrderedRing R] [DecidableEq n]
    {M : Matrix n n R} (hM : M.PosSemidef) (k : ℕ) :
    PosSemidef (M ^ k) :=
  match k with
  | 0 => .one
            /-
              n : Type u_2
              R : Type u_3
              inst✝⁵ : Fintype n
              inst✝⁴ : CommRing R
              inst✝³ : PartialOrder R
              inst✝² : StarRing R
              inst✝¹ : StarOrderedRing R
              inst✝ : DecidableEq n
              M : Matrix n n R
              hM : M.PosSemidef
              k : Nat
              ⊢ (HPow.hPow M 1).PosSemidef
            -/
  | 1 => by simpa using hM
            /-
              🎉 no goals
            -/
  | (k + 2) => by
    /-
      n : Type u_2
      R : Type u_3
      inst✝⁵ : Fintype n
      inst✝⁴ : CommRing R
      inst✝³ : PartialOrder R
      inst✝² : StarRing R
      inst✝¹ : StarOrderedRing R
      inst✝ : DecidableEq n
      M : Matrix n n R
      hM : M.PosSemidef
      k✝ k : Nat
      ⊢ (HPow.hPow M (HAdd.hAdd k 2)).PosSemidef
    -/
    rw [pow_succ, pow_succ']
    /-
      n : Type u_2
      R : Type u_3
      inst✝⁵ : Fintype n
      inst✝⁴ : CommRing R
      inst✝³ : PartialOrder R
      inst✝² : StarRing R
      inst✝¹ : StarOrderedRing R
      inst✝ : DecidableEq n
      M : Matrix n n R
      hM : M.PosSemidef
      k✝ k : Nat
      ⊢ (HMul.hMul (HMul.hMul M (HPow.hPow M k)) M).PosSemidef
    -/
    simpa only [hM.isHermitian.eq] using (hM.pow k).mul_mul_conjTranspose_same M
    /-
      🎉 no goals
    -/


protected lemma inv [DecidableEq n] {M : Matrix n n R} (hM : M.PosSemidef) : M⁻¹.PosSemidef := by
  /-
    n : Type u_2
    R : Type u_3
    inst✝⁴ : Fintype n
    inst✝³ : CommRing R
    inst✝² : PartialOrder R
    inst✝¹ : StarRing R
    inst✝ : DecidableEq n
    M : Matrix n n R
    hM : M.PosSemidef
    ⊢ (Inv.inv M).PosSemidef
  -/
  by_cases h : IsUnit M.det
    /-
      case pos
      n : Type u_2
      R : Type u_3
      inst✝⁴ : Fintype n
      inst✝³ : CommRing R
      inst✝² : PartialOrder R
      inst✝¹ : StarRing R
      inst✝ : DecidableEq n
      M : Matrix n n R
      hM : M.PosSemidef
      h : IsUnit M.det
      ⊢ (Inv.inv M).PosSemidef
    -/
  · have := (conjTranspose_mul_mul_same hM M⁻¹).conjTranspose
    /-
      case pos
      n : Type u_2
      R : Type u_3
      inst✝⁴ : Fintype n
      inst✝³ : CommRing R
      inst✝² : PartialOrder R
      inst✝¹ : StarRing R
      inst✝ : DecidableEq n
      M : Matrix n n R
      hM : M.PosSemidef
      h : IsUnit M.det
      this : (HMul.hMul (HMul.hMul (Inv.inv M).conjTranspose M) (Inv.inv M)).conjTra …
      ⊢ (Inv.inv M).PosSemidef
    -/
    rwa [mul_nonsing_inv_cancel_right _ _ h, conjTranspose_conjTranspose] at this
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Type u_2
      R : Type u_3
      inst✝⁴ : Fintype n
      inst✝³ : CommRing R
      inst✝² : PartialOrder R
      inst✝¹ : StarRing R
      inst✝ : DecidableEq n
      M : Matrix n n R
      hM : M.PosSemidef
      h : Not (IsUnit M.det)
      ⊢ (Inv.inv M).PosSemidef
    -/
  · rw [nonsing_inv_apply_not_isUnit _ h]
    /-
      case neg
      n : Type u_2
      R : Type u_3
      inst✝⁴ : Fintype n
      inst✝³ : CommRing R
      inst✝² : PartialOrder R
      inst✝¹ : StarRing R
      inst✝ : DecidableEq n
      M : Matrix n n R
      hM : M.PosSemidef
      h : Not (IsUnit M.det)
      ⊢ Matrix.PosSemidef 0
    -/
    exact .zero
    /-
      🎉 no goals
    -/


protected lemma zpow [StarOrderedRing R] [DecidableEq n]
    {M : Matrix n n R} (hM : M.PosSemidef) (z : ℤ) :
    (M ^ z).PosSemidef := by
  /-
    n : Type u_2
    R : Type u_3
    inst✝⁵ : Fintype n
    inst✝⁴ : CommRing R
    inst✝³ : PartialOrder R
    inst✝² : StarRing R
    inst✝¹ : StarOrderedRing R
    inst✝ : DecidableEq n
    M : Matrix n n R
    hM : M.PosSemidef
    z : Int
    ⊢ (HPow.hPow M z).PosSemidef
  -/
  obtain ⟨n, rfl | rfl⟩ := z.eq_nat_or_neg
    /-
      case intro.inl
      n✝ : Type u_2
      R : Type u_3
      inst✝⁵ : Fintype n✝
      inst✝⁴ : CommRing R
      inst✝³ : PartialOrder R
      inst✝² : StarRing R
      inst✝¹ : StarOrderedRing R
      inst✝ : DecidableEq n✝
      M : Matrix n✝ n✝ R
      hM : M.PosSemidef
      n : Nat
      ⊢ (HPow.hPow M ↑n).PosSemidef
    -/
  · simpa using hM.pow n
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      n✝ : Type u_2
      R : Type u_3
      inst✝⁵ : Fintype n✝
      inst✝⁴ : CommRing R
      inst✝³ : PartialOrder R
      inst✝² : StarRing R
      inst✝¹ : StarOrderedRing R
      inst✝ : DecidableEq n✝
      M : Matrix n✝ n✝ R
      hM : M.PosSemidef
      n : Nat
      ⊢ (HPow.hPow M (Neg.neg ↑n)).PosSemidef
    -/
  · simpa using (hM.pow n).inv
    /-
      🎉 no goals
    -/


protected lemma add [AddLeftMono R] {A : Matrix m m R} {B : Matrix m m R}
    (hA : A.PosSemidef) (hB : B.PosSemidef) : (A + B).PosSemidef :=
  ⟨hA.isHermitian.add hB.isHermitian, fun x => by
    /-
      m : Type u_1
      R : Type u_3
      inst✝⁴ : Fintype m
      inst✝³ : CommRing R
      inst✝² : PartialOrder R
      inst✝¹ : StarRing R
      inst✝ : AddLeftMono R
      A B : Matrix m m R
      hA : A.PosSemidef
      hB : B.PosSemidef
      x : m → R
      ⊢ LE.le 0 (dotProduct (Star.star x) ((HAdd.hAdd A B).mulVec x))
    -/
    rw [add_mulVec, dotProduct_add]
    /-
      m : Type u_1
      R : Type u_3
      inst✝⁴ : Fintype m
      inst✝³ : CommRing R
      inst✝² : PartialOrder R
      inst✝¹ : StarRing R
      inst✝ : AddLeftMono R
      A B : Matrix m m R
      hA : A.PosSemidef
      hB : B.PosSemidef
      x : m → R
      ⊢ LE.le 0 (HAdd.hAdd (dotProduct (Star.star x) (A.mulVec x)) (dotProduct (Star …
    -/
    exact add_nonneg (hA.2 x) (hB.2 x)⟩
    /-
      🎉 no goals
    -/


/-- The eigenvalues of a positive semi-definite matrix are non-negative -/
lemma eigenvalues_nonneg [DecidableEq n] {A : Matrix n n 𝕜}
    (hA : Matrix.PosSemidef A) (i : n) : 0 ≤ hA.1.eigenvalues i :=
  (hA.re_dotProduct_nonneg _).trans_eq (hA.1.eigenvalues_eq _).symm


/-- The positive semidefinite square root of a positive semidefinite matrix -/
noncomputable def sqrt : Matrix n n 𝕜 :=
  hA.1.eigenvectorUnitary.1 * diagonal ((↑) ∘ Real.sqrt ∘ hA.1.eigenvalues) *
  (star hA.1.eigenvectorUnitary : Matrix n n 𝕜)


open Lean PrettyPrinter.Delaborator SubExpr in
/-- Custom elaborator to produce output like `(_ : PosSemidef A).sqrt` in the goal view. -/
@[app_delab Matrix.PosSemidef.sqrt]
def delabSqrt : Delab :=
  whenPPOption getPPNotation <|
  whenNotPPOption getPPAnalysisSkip <|
  withOverApp 7 <|
  withOptionAtCurrPos `pp.analysis.skip true do
    let e ← getExpr
    guard <| e.isAppOfArity ``Matrix.PosSemidef.sqrt 7
    let optionsPerPos ← withNaryArg 6 do
      return (← read).optionsPerPos.setBool (← getPos) `pp.proofs.withType true
    withTheReader Context ({· with optionsPerPos}) delab


lemma posSemidef_sqrt : PosSemidef hA.sqrt := by
  /-
    n : Type u_2
    𝕜 : Type u_4
    inst✝² : Fintype n
    inst✝¹ : RCLike 𝕜
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    hA : A.PosSemidef
    ⊢ hA.sqrt.PosSemidef
  -/
  apply PosSemidef.mul_mul_conjTranspose_same
  /-
    case hA
    n : Type u_2
    𝕜 : Type u_4
    inst✝² : Fintype n
    inst✝¹ : RCLike 𝕜
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    hA : A.PosSemidef
    ⊢ (Matrix.diagonal (Function.comp RCLike.ofReal (Function.comp Real.sqrt ⋯.eig …
  -/
  refine posSemidef_diagonal_iff.mpr fun i ↦ ?_
  /-
    case hA
    n : Type u_2
    𝕜 : Type u_4
    inst✝² : Fintype n
    inst✝¹ : RCLike 𝕜
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    hA : A.PosSemidef
    i : n
    ⊢ LE.le 0 (Function.comp RCLike.ofReal (Function.comp Real.sqrt ⋯.eigenvalues) …
  -/
  rw [Function.comp_apply, RCLike.nonneg_iff]
  /-
    case hA
    n : Type u_2
    𝕜 : Type u_4
    inst✝² : Fintype n
    inst✝¹ : RCLike 𝕜
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    hA : A.PosSemidef
    i : n
    ⊢ And (LE.le 0 (RCLike.re ↑(Function.comp Real.sqrt ⋯.eigenvalues i))) (Eq (RC …
  -/
  constructor
    /-
      case hA.left
      n : Type u_2
      𝕜 : Type u_4
      inst✝² : Fintype n
      inst✝¹ : RCLike 𝕜
      inst✝ : DecidableEq n
      A : Matrix n n 𝕜
      hA : A.PosSemidef
      i : n
      ⊢ LE.le 0 (RCLike.re ↑(Function.comp Real.sqrt ⋯.eigenvalues i))
    -/
  · simp only [RCLike.ofReal_re]
    /-
      case hA.left
      n : Type u_2
      𝕜 : Type u_4
      inst✝² : Fintype n
      inst✝¹ : RCLike 𝕜
      inst✝ : DecidableEq n
      A : Matrix n n 𝕜
      hA : A.PosSemidef
      i : n
      ⊢ LE.le 0 (Function.comp Real.sqrt ⋯.eigenvalues i)
    -/
    exact Real.sqrt_nonneg _
    /-
      🎉 no goals
    -/
    /-
      case hA.right
      n : Type u_2
      𝕜 : Type u_4
      inst✝² : Fintype n
      inst✝¹ : RCLike 𝕜
      inst✝ : DecidableEq n
      A : Matrix n n 𝕜
      hA : A.PosSemidef
      i : n
      ⊢ Eq (RCLike.im ↑(Function.comp Real.sqrt ⋯.eigenvalues i)) 0
    -/
  · simp only [RCLike.ofReal_im]
    /-
      🎉 no goals
    -/


@[simp]
lemma sq_sqrt : hA.sqrt ^ 2 = A := by
  /-
    n : Type u_2
    𝕜 : Type u_4
    inst✝² : Fintype n
    inst✝¹ : RCLike 𝕜
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    hA : A.PosSemidef
    ⊢ Eq (HPow.hPow hA.sqrt 2) A
  -/
  let C : Matrix n n 𝕜 := hA.1.eigenvectorUnitary
  /-
    n : Type u_2
    𝕜 : Type u_4
    inst✝² : Fintype n
    inst✝¹ : RCLike 𝕜
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    hA : A.PosSemidef
    C : Matrix n n 𝕜 := ↑⋯.eigenvectorUnitary
    ⊢ Eq (HPow.hPow hA.sqrt 2) A
  -/
  let E := diagonal ((↑) ∘ Real.sqrt ∘ hA.1.eigenvalues : n → 𝕜)
  suffices C * (E * (star C * C) * E) * star C = A by
    rw [Matrix.PosSemidef.sqrt, pow_two]
    simpa only [← mul_assoc] using this
  have : E * E = diagonal ((↑) ∘ hA.1.eigenvalues) := by
    rw [diagonal_mul_diagonal]
    congr! with v
    simp [← pow_two, ← RCLike.ofReal_pow, Real.sq_sqrt (hA.eigenvalues_nonneg v)]
  /-
    n : Type u_2
    𝕜 : Type u_4
    inst✝² : Fintype n
    inst✝¹ : RCLike 𝕜
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    hA : A.PosSemidef
    C : Matrix n n 𝕜 := ↑⋯.eigenvectorUnitary
    E : Matrix n n 𝕜 := Matrix.diagonal (Function.comp RCLike.ofReal (Function.com …
    this : Eq (HMul.hMul E E) (Matrix.diagonal (Function.comp RCLike.ofReal ⋯.eige …
    ⊢ Eq (HMul.hMul (HMul.hMul C (HMul.hMul (HMul.hMul E (HMul.hMul (Star.star C)  …
  -/
  simpa [C, this] using hA.1.spectral_theorem.symm
  /-
    🎉 no goals
  -/


@[simp]
                                                  /-
                                                    n : Type u_2
                                                    𝕜 : Type u_4
                                                    inst✝² : Fintype n
                                                    inst✝¹ : RCLike 𝕜
                                                    inst✝ : DecidableEq n
                                                    A : Matrix n n 𝕜
                                                    hA : A.PosSemidef
                                                    ⊢ Eq (HMul.hMul hA.sqrt hA.sqrt) A
                                                  -/
lemma sqrt_mul_self : hA.sqrt * hA.sqrt = A := by rw [← pow_two, sq_sqrt]
                                                  /-
                                                    🎉 no goals
                                                  -/


include hA in
lemma eq_of_sq_eq_sq {B : Matrix n n 𝕜} (hB : PosSemidef B) (hAB : A ^ 2 = B ^ 2) : A = B := by
  /- This is deceptively hard, much more difficult than the positive *definite* case. We follow a
  clever proof due to Koeber and Schäfer. The idea is that if `A ≠ B`, then `A - B` has a nonzero
  real eigenvalue, with eigenvector `v`. Then a manipulation using the identity
  `A ^ 2 - B ^ 2 = A * (A - B) + (A - B) * B` leads to the conclusion that
  `⟨v, A v⟩ + ⟨v, B v⟩ = 0`. Since `A, B` are positive semidefinite, both terms must be zero. Thus
  `⟨v, (A - B) v⟩ = 0`, but this is a nonzero scalar multiple of `⟨v, v⟩`, contradiction. -/
  /-
    n : Type u_2
    𝕜 : Type u_4
    inst✝² : Fintype n
    inst✝¹ : RCLike 𝕜
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    hA : A.PosSemidef
    B : Matrix n n 𝕜
    hB : B.PosSemidef
    hAB : Eq (HPow.hPow A 2) (HPow.hPow B 2)
    ⊢ Eq A B
  -/
  by_contra h_ne
  /-
    n : Type u_2
    𝕜 : Type u_4
    inst✝² : Fintype n
    inst✝¹ : RCLike 𝕜
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    hA : A.PosSemidef
    B : Matrix n n 𝕜
    hB : B.PosSemidef
    hAB : Eq (HPow.hPow A 2) (HPow.hPow B 2)
    h_ne : Not (Eq A B)
    ⊢ False
  -/
  let ⟨v, t, ht, hv, hv'⟩ := (hA.1.sub hB.1).exists_eigenvector_of_ne_zero (sub_ne_zero.mpr h_ne)
  have h_sum : 0 = t * (star v ⬝ᵥ A *ᵥ v + star v ⬝ᵥ B *ᵥ v) := calc
    0 = star v ⬝ᵥ (A ^ 2 - B ^ 2) *ᵥ v := by rw [hAB, sub_self, zero_mulVec, dotProduct_zero]
    _ = star v ⬝ᵥ A *ᵥ (A - B) *ᵥ v + star v ⬝ᵥ (A - B) *ᵥ B *ᵥ v := by
      rw [mulVec_mulVec, mulVec_mulVec, ← dotProduct_add, ← add_mulVec, mul_sub, sub_mul,
        add_sub, sub_add_cancel, pow_two, pow_two]
    _ = t * (star v ⬝ᵥ A *ᵥ v) + (star v) ᵥ* (A - B)ᴴ ⬝ᵥ B *ᵥ v := by
      rw [hv', mulVec_smul, dotProduct_smul, RCLike.real_smul_eq_coe_mul,
        dotProduct_mulVec _ (A - B), hA.1.sub hB.1]
    _ = t * (star v ⬝ᵥ A *ᵥ v + star v ⬝ᵥ B *ᵥ v) := by
      simp_rw [← star_mulVec, hv', mul_add, ← RCLike.real_smul_eq_coe_mul, ← smul_dotProduct]
      congr 2 with i
      simp only [Pi.star_apply, Pi.smul_apply, RCLike.real_smul_eq_coe_mul, star_mul',
        RCLike.star_def, RCLike.conj_ofReal]
  replace h_sum : star v ⬝ᵥ A *ᵥ v + star v ⬝ᵥ B *ᵥ v = 0 := by
    rw [eq_comm, ← mul_zero (t : 𝕜)] at h_sum
    exact mul_left_cancel₀ (RCLike.ofReal_ne_zero.mpr ht) h_sum
  have h_van : star v ⬝ᵥ A *ᵥ v = 0 ∧ star v ⬝ᵥ B *ᵥ v = 0 := by
    refine ⟨le_antisymm ?_ (hA.2 v), le_antisymm ?_ (hB.2 v)⟩
    · rw [add_comm, add_eq_zero_iff_eq_neg] at h_sum
      simpa only [h_sum, neg_nonneg] using hB.2 v
    · simpa only [add_eq_zero_iff_eq_neg.mp h_sum, neg_nonneg] using hA.2 v
  have aux : star v ⬝ᵥ (A - B) *ᵥ v = 0 := by
    rw [sub_mulVec, dotProduct_sub, h_van.1, h_van.2, sub_zero]
  /-
    n : Type u_2
    𝕜 : Type u_4
    inst✝² : Fintype n
    inst✝¹ : RCLike 𝕜
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    hA : A.PosSemidef
    B : Matrix n n 𝕜
    hB : B.PosSemidef
    hAB : Eq (HPow.hPow A 2) (HPow.hPow B 2)
    h_ne : Not (Eq A B)
    v : n → 𝕜
    t : Real
    ht : Ne t 0
    hv : Ne v 0
    hv' : Eq ((HSub.hSub A B).mulVec v) (HSMul.hSMul t v)
    h_sum : Eq (HAdd.hAdd (dotProduct (Star.star v) (A.mulVec v)) (dotProduct (Sta …
    h_van : And (Eq (dotProduct (Star.star v) (A.mulVec v)) 0) (Eq (dotProduct (St …
    aux : Eq (dotProduct (Star.star v) ((HSub.hSub A B).mulVec v)) 0
    ⊢ False
  -/
  rw [hv', dotProduct_smul, RCLike.real_smul_eq_coe_mul, ← mul_zero ↑t] at aux
  exact hv <| dotProduct_star_self_eq_zero.mp <| mul_left_cancel₀
    (RCLike.ofReal_ne_zero.mpr ht) aux


lemma sqrt_sq : (hA.pow 2 : PosSemidef (A ^ 2)).sqrt = A :=
  (hA.pow 2).posSemidef_sqrt.eq_of_sq_eq_sq hA (hA.pow 2).sq_sqrt


include hA in
lemma eq_sqrt_of_sq_eq {B : Matrix n n 𝕜} (hB : PosSemidef B) (hAB : A ^ 2 = B) : A = hB.sqrt := by
  /-
    n : Type u_2
    𝕜 : Type u_4
    inst✝² : Fintype n
    inst✝¹ : RCLike 𝕜
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    hA : A.PosSemidef
    B : Matrix n n 𝕜
    hB : B.PosSemidef
    hAB : Eq (HPow.hPow A 2) B
    ⊢ Eq A hB.sqrt
  -/
  subst B
  /-
    n : Type u_2
    𝕜 : Type u_4
    inst✝² : Fintype n
    inst✝¹ : RCLike 𝕜
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    hA : A.PosSemidef
    hB : (HPow.hPow A 2).PosSemidef
    ⊢ Eq A hB.sqrt
  -/
  rw [hA.sqrt_sq]
  /-
    🎉 no goals
  -/


@[simp]
theorem posSemidef_submatrix_equiv {M : Matrix n n R} (e : m ≃ n) :
    (M.submatrix e e).PosSemidef ↔ M.PosSemidef :=
               /-
                 m : Type u_1
                 n : Type u_2
                 R : Type u_3
                 inst✝⁴ : Fintype m
                 inst✝³ : Fintype n
                 inst✝² : CommRing R
                 inst✝¹ : PartialOrder R
                 inst✝ : StarRing R
                 M : Matrix n n R
                 e : Equiv m n
                 h : (M.submatrix ⇑e ⇑e).PosSemidef
                 ⊢ M.PosSemidef
               -/
  ⟨fun h => by simpa using h.submatrix e.symm, fun h => h.submatrix _⟩
               /-
                 🎉 no goals
               -/


/-- The conjugate transpose of a matrix multiplied by the matrix is positive semidefinite -/
theorem posSemidef_conjTranspose_mul_self [StarOrderedRing R] (A : Matrix m n R) :
    PosSemidef (Aᴴ * A) := by
  /-
    m : Type u_1
    n : Type u_2
    R : Type u_3
    inst✝⁵ : Fintype m
    inst✝⁴ : Fintype n
    inst✝³ : CommRing R
    inst✝² : PartialOrder R
    inst✝¹ : StarRing R
    inst✝ : StarOrderedRing R
    A : Matrix m n R
    ⊢ (HMul.hMul A.conjTranspose A).PosSemidef
  -/
  refine ⟨isHermitian_transpose_mul_self _, fun x => ?_⟩
  /-
    m : Type u_1
    n : Type u_2
    R : Type u_3
    inst✝⁵ : Fintype m
    inst✝⁴ : Fintype n
    inst✝³ : CommRing R
    inst✝² : PartialOrder R
    inst✝¹ : StarRing R
    inst✝ : StarOrderedRing R
    A : Matrix m n R
    x : n → R
    ⊢ LE.le 0 (dotProduct (Star.star x) ((HMul.hMul A.conjTranspose A).mulVec x))
  -/
  rw [← mulVec_mulVec, dotProduct_mulVec, vecMul_conjTranspose, star_star]
  /-
    m : Type u_1
    n : Type u_2
    R : Type u_3
    inst✝⁵ : Fintype m
    inst✝⁴ : Fintype n
    inst✝³ : CommRing R
    inst✝² : PartialOrder R
    inst✝¹ : StarRing R
    inst✝ : StarOrderedRing R
    A : Matrix m n R
    x : n → R
    ⊢ LE.le 0 (dotProduct (Star.star (A.mulVec x)) (A.mulVec x))
  -/
  exact Finset.sum_nonneg fun i _ => star_mul_self_nonneg _
  /-
    🎉 no goals
  -/


/-- A matrix multiplied by its conjugate transpose is positive semidefinite -/
theorem posSemidef_self_mul_conjTranspose [StarOrderedRing R] (A : Matrix m n R) :
    PosSemidef (A * Aᴴ) := by
  /-
    m : Type u_1
    n : Type u_2
    R : Type u_3
    inst✝⁵ : Fintype m
    inst✝⁴ : Fintype n
    inst✝³ : CommRing R
    inst✝² : PartialOrder R
    inst✝¹ : StarRing R
    inst✝ : StarOrderedRing R
    A : Matrix m n R
    ⊢ (HMul.hMul A A.conjTranspose).PosSemidef
  -/
  simpa only [conjTranspose_conjTranspose] using posSemidef_conjTranspose_mul_self Aᴴ
  /-
    🎉 no goals
  -/


lemma eigenvalues_conjTranspose_mul_self_nonneg (A : Matrix m n 𝕜) [DecidableEq n] (i : n) :
    0 ≤ (isHermitian_transpose_mul_self A).eigenvalues i :=
  (posSemidef_conjTranspose_mul_self _).eigenvalues_nonneg _


lemma eigenvalues_self_mul_conjTranspose_nonneg (A : Matrix m n 𝕜) [DecidableEq m] (i : m) :
    0 ≤ (isHermitian_mul_conjTranspose_self A).eigenvalues i :=
  (posSemidef_self_mul_conjTranspose _).eigenvalues_nonneg _


/-- A matrix is positive semidefinite if and only if it has the form `Bᴴ * B` for some `B`. -/
lemma posSemidef_iff_eq_transpose_mul_self {A : Matrix n n 𝕜} :
    PosSemidef A ↔ ∃ (B : Matrix n n 𝕜), A = Bᴴ * B := by
  classical
  refine ⟨fun hA ↦ ⟨hA.sqrt, ?_⟩, fun ⟨B, hB⟩ ↦ (hB ▸ posSemidef_conjTranspose_mul_self B)⟩
  simp_rw [← PosSemidef.sq_sqrt hA, pow_two]
  rw [hA.posSemidef_sqrt.1]


lemma IsHermitian.posSemidef_of_eigenvalues_nonneg [DecidableEq n] {A : Matrix n n 𝕜}
    (hA : IsHermitian A) (h : ∀ i : n, 0 ≤ hA.eigenvalues i) : PosSemidef A := by
  /-
    n : Type u_2
    𝕜 : Type u_4
    inst✝² : Fintype n
    inst✝¹ : RCLike 𝕜
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    hA : A.IsHermitian
    h : ∀ (i : n), LE.le 0 (hA.eigenvalues i)
    ⊢ A.PosSemidef
  -/
  rw [hA.spectral_theorem]
  /-
    n : Type u_2
    𝕜 : Type u_4
    inst✝² : Fintype n
    inst✝¹ : RCLike 𝕜
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    hA : A.IsHermitian
    h : ∀ (i : n), LE.le 0 (hA.eigenvalues i)
    ⊢ (HMul.hMul (HMul.hMul (↑hA.eigenvectorUnitary) (Matrix.diagonal (Function.co …
  -/
  refine (posSemidef_diagonal_iff.mpr ?_).mul_mul_conjTranspose_same _
  /-
    n : Type u_2
    𝕜 : Type u_4
    inst✝² : Fintype n
    inst✝¹ : RCLike 𝕜
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    hA : A.IsHermitian
    h : ∀ (i : n), LE.le 0 (hA.eigenvalues i)
    ⊢ ∀ (i : n), LE.le 0 (Function.comp RCLike.ofReal hA.eigenvalues i)
  -/
  simpa using h
  /-
    🎉 no goals
  -/


/-- For `A` positive semidefinite, we have `x⋆ A x = 0` iff `A x = 0`. -/
theorem PosSemidef.dotProduct_mulVec_zero_iff
    {A : Matrix n n 𝕜} (hA : PosSemidef A) (x : n → 𝕜) :
    star x ⬝ᵥ A *ᵥ x = 0 ↔ A *ᵥ x = 0 := by
  /-
    n : Type u_2
    𝕜 : Type u_4
    inst✝¹ : Fintype n
    inst✝ : RCLike 𝕜
    A : Matrix n n 𝕜
    hA : A.PosSemidef
    x : n → 𝕜
    ⊢ Iff (Eq (dotProduct (Star.star x) (A.mulVec x)) 0) (Eq (A.mulVec x) 0)
  -/
  constructor
    /-
      case mp
      n : Type u_2
      𝕜 : Type u_4
      inst✝¹ : Fintype n
      inst✝ : RCLike 𝕜
      A : Matrix n n 𝕜
      hA : A.PosSemidef
      x : n → 𝕜
      ⊢ Eq (dotProduct (Star.star x) (A.mulVec x)) 0 → Eq (A.mulVec x) 0
    -/
  · obtain ⟨B, rfl⟩ := posSemidef_iff_eq_transpose_mul_self.mp hA
    rw [← Matrix.mulVec_mulVec, dotProduct_mulVec,
      vecMul_conjTranspose, star_star, dotProduct_star_self_eq_zero]
    /-
      case mp.intro
      n : Type u_2
      𝕜 : Type u_4
      inst✝¹ : Fintype n
      inst✝ : RCLike 𝕜
      x : n → 𝕜
      B : Matrix n n 𝕜
      hA : (HMul.hMul B.conjTranspose B).PosSemidef
      ⊢ Eq (B.mulVec x) 0 → Eq (B.conjTranspose.mulVec (B.mulVec x)) 0
    -/
    intro h0
    /-
      case mp.intro
      n : Type u_2
      𝕜 : Type u_4
      inst✝¹ : Fintype n
      inst✝ : RCLike 𝕜
      x : n → 𝕜
      B : Matrix n n 𝕜
      hA : (HMul.hMul B.conjTranspose B).PosSemidef
      h0 : Eq (B.mulVec x) 0
      ⊢ Eq (B.conjTranspose.mulVec (B.mulVec x)) 0
    -/
    rw [h0, mulVec_zero]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      n : Type u_2
      𝕜 : Type u_4
      inst✝¹ : Fintype n
      inst✝ : RCLike 𝕜
      A : Matrix n n 𝕜
      hA : A.PosSemidef
      x : n → 𝕜
      ⊢ Eq (A.mulVec x) 0 → Eq (dotProduct (Star.star x) (A.mulVec x)) 0
    -/
  · intro h0
    /-
      case mpr
      n : Type u_2
      𝕜 : Type u_4
      inst✝¹ : Fintype n
      inst✝ : RCLike 𝕜
      A : Matrix n n 𝕜
      hA : A.PosSemidef
      x : n → 𝕜
      h0 : Eq (A.mulVec x) 0
      ⊢ Eq (dotProduct (Star.star x) (A.mulVec x)) 0
    -/
    rw [h0, dotProduct_zero]
    /-
      🎉 no goals
    -/


/-- For `A` positive semidefinite, we have `x⋆ A x = 0` iff `A x = 0` (linear maps version). -/
theorem PosSemidef.toLinearMap₂'_zero_iff [DecidableEq n]
    {A : Matrix n n 𝕜} (hA : PosSemidef A) (x : n → 𝕜) :
    Matrix.toLinearMap₂' 𝕜 A (star x) x = 0 ↔ Matrix.toLin' A x = 0 := by
  /-
    n : Type u_2
    𝕜 : Type u_4
    inst✝² : Fintype n
    inst✝¹ : RCLike 𝕜
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    hA : A.PosSemidef
    x : n → 𝕜
    ⊢ Iff (Eq ((((Matrix.toLinearMap₂' 𝕜) A) (Star.star x)) x) 0) (Eq ((Matrix.toL …
  -/
  simpa only [toLinearMap₂'_apply', toLin'_apply] using hA.dotProduct_mulVec_zero_iff x
  /-
    🎉 no goals
  -/


/-- A matrix `M : Matrix n n R` is positive definite if it is hermitian
   and `xᴴMx` is greater than zero for all nonzero `x`. -/
def PosDef (M : Matrix n n R) :=
  M.IsHermitian ∧ ∀ x : n → R, x ≠ 0 → 0 < dotProduct (star x) (M *ᵥ x)


theorem isHermitian {M : Matrix n n R} (hM : M.PosDef) : M.IsHermitian :=
  hM.1


theorem re_dotProduct_pos {M : Matrix n n 𝕜} (hM : M.PosDef) {x : n → 𝕜} (hx : x ≠ 0) :
    0 < RCLike.re (dotProduct (star x) (M *ᵥ x)) :=
  RCLike.pos_iff.mp (hM.2 _ hx) |>.1


theorem posSemidef {M : Matrix n n R} (hM : M.PosDef) : M.PosSemidef := by
  /-
    n : Type u_2
    R : Type u_3
    inst✝³ : Fintype n
    inst✝² : CommRing R
    inst✝¹ : PartialOrder R
    inst✝ : StarRing R
    M : Matrix n n R
    hM : M.PosDef
    ⊢ M.PosSemidef
  -/
  refine ⟨hM.1, ?_⟩
  /-
    n : Type u_2
    R : Type u_3
    inst✝³ : Fintype n
    inst✝² : CommRing R
    inst✝¹ : PartialOrder R
    inst✝ : StarRing R
    M : Matrix n n R
    hM : M.PosDef
    ⊢ ∀ (x : n → R), LE.le 0 (dotProduct (Star.star x) (M.mulVec x))
  -/
  intro x
  /-
    n : Type u_2
    R : Type u_3
    inst✝³ : Fintype n
    inst✝² : CommRing R
    inst✝¹ : PartialOrder R
    inst✝ : StarRing R
    M : Matrix n n R
    hM : M.PosDef
    x : n → R
    ⊢ LE.le 0 (dotProduct (Star.star x) (M.mulVec x))
  -/
  by_cases hx : x = 0
    /-
      case pos
      n : Type u_2
      R : Type u_3
      inst✝³ : Fintype n
      inst✝² : CommRing R
      inst✝¹ : PartialOrder R
      inst✝ : StarRing R
      M : Matrix n n R
      hM : M.PosDef
      x : n → R
      hx : Eq x 0
      ⊢ LE.le 0 (dotProduct (Star.star x) (M.mulVec x))
    -/
  · simp only [hx, zero_dotProduct, star_zero, RCLike.zero_re']
    /-
      case pos
      n : Type u_2
      R : Type u_3
      inst✝³ : Fintype n
      inst✝² : CommRing R
      inst✝¹ : PartialOrder R
      inst✝ : StarRing R
      M : Matrix n n R
      hM : M.PosDef
      x : n → R
      hx : Eq x 0
      ⊢ LE.le 0 0
    -/
    exact le_rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Type u_2
      R : Type u_3
      inst✝³ : Fintype n
      inst✝² : CommRing R
      inst✝¹ : PartialOrder R
      inst✝ : StarRing R
      M : Matrix n n R
      hM : M.PosDef
      x : n → R
      hx : Not (Eq x 0)
      ⊢ LE.le 0 (dotProduct (Star.star x) (M.mulVec x))
    -/
  · exact le_of_lt (hM.2 x hx)
    /-
      🎉 no goals
    -/


theorem transpose {M : Matrix n n R} (hM : M.PosDef) : Mᵀ.PosDef := by
  /-
    n : Type u_2
    R : Type u_3
    inst✝³ : Fintype n
    inst✝² : CommRing R
    inst✝¹ : PartialOrder R
    inst✝ : StarRing R
    M : Matrix n n R
    hM : M.PosDef
    ⊢ M.transpose.PosDef
  -/
  refine ⟨IsHermitian.transpose hM.1, fun x hx => ?_⟩
  /-
    n : Type u_2
    R : Type u_3
    inst✝³ : Fintype n
    inst✝² : CommRing R
    inst✝¹ : PartialOrder R
    inst✝ : StarRing R
    M : Matrix n n R
    hM : M.PosDef
    x : n → R
    hx : Ne x 0
    ⊢ LT.lt 0 (dotProduct (Star.star x) (M.transpose.mulVec x))
  -/
  convert hM.2 (star x) (star_ne_zero.2 hx) using 1
  /-
    case h.e'_4
    n : Type u_2
    R : Type u_3
    inst✝³ : Fintype n
    inst✝² : CommRing R
    inst✝¹ : PartialOrder R
    inst✝ : StarRing R
    M : Matrix n n R
    hM : M.PosDef
    x : n → R
    hx : Ne x 0
    ⊢ Eq (dotProduct (Star.star x) (M.transpose.mulVec x)) (dotProduct (Star.star  …
  -/
  rw [mulVec_transpose, dotProduct_mulVec, star_star, dotProduct_comm]
  /-
    🎉 no goals
  -/


protected theorem diagonal [StarOrderedRing R] [DecidableEq n] [NoZeroDivisors R]
    {d : n → R} (h : ∀ i, 0 < d i) :
    PosDef (diagonal d) :=
  ⟨isHermitian_diagonal_of_self_adjoint _ <| funext fun i => IsSelfAdjoint.of_nonneg (h i).le,
    fun x hx => by
      /-
        n : Type u_2
        R : Type u_3
        inst✝⁶ : Fintype n
        inst✝⁵ : CommRing R
        inst✝⁴ : PartialOrder R
        inst✝³ : StarRing R
        inst✝² : StarOrderedRing R
        inst✝¹ : DecidableEq n
        inst✝ : NoZeroDivisors R
        d : n → R
        h : ∀ (i : n), LT.lt 0 (d i)
        x : n → R
        hx : Ne x 0
        ⊢ LT.lt 0 (dotProduct (Star.star x) ((Matrix.diagonal d).mulVec x))
      -/
      refine Fintype.sum_pos ?_
      /-
        n : Type u_2
        R : Type u_3
        inst✝⁶ : Fintype n
        inst✝⁵ : CommRing R
        inst✝⁴ : PartialOrder R
        inst✝³ : StarRing R
        inst✝² : StarOrderedRing R
        inst✝¹ : DecidableEq n
        inst✝ : NoZeroDivisors R
        d : n → R
        h : ∀ (i : n), LT.lt 0 (d i)
        x : n → R
        hx : Ne x 0
        ⊢ LT.lt 0 fun i => HMul.hMul (Star.star x i) ((Matrix.diagonal d).mulVec x i)
      -/
      simp_rw [mulVec_diagonal, ← mul_assoc, Pi.lt_def]
      /-
        n : Type u_2
        R : Type u_3
        inst✝⁶ : Fintype n
        inst✝⁵ : CommRing R
        inst✝⁴ : PartialOrder R
        inst✝³ : StarRing R
        inst✝² : StarOrderedRing R
        inst✝¹ : DecidableEq n
        inst✝ : NoZeroDivisors R
        d : n → R
        h : ∀ (i : n), LT.lt 0 (d i)
        x : n → R
        hx : Ne x 0
        ⊢ And (LE.le 0 fun i => HMul.hMul (HMul.hMul (Star.star x i) (d i)) (x i)) (Ex …
      -/
      obtain ⟨i, hi⟩ := Function.ne_iff.mp hx
      exact ⟨fun i => conjugate_nonneg (h i).le _,
        i, conjugate_pos (h _) (isRegular_of_ne_zero hi)⟩⟩


@[simp]
theorem _root_.Matrix.posDef_diagonal_iff
    [StarOrderedRing R] [DecidableEq n] [NoZeroDivisors R] [Nontrivial R] {d : n → R} :
    PosDef (diagonal d) ↔ ∀ i, 0 < d i := by
  /-
    n : Type u_2
    R : Type u_3
    inst✝⁷ : Fintype n
    inst✝⁶ : CommRing R
    inst✝⁵ : PartialOrder R
    inst✝⁴ : StarRing R
    inst✝³ : StarOrderedRing R
    inst✝² : DecidableEq n
    inst✝¹ : NoZeroDivisors R
    inst✝ : Nontrivial R
    d : n → R
    ⊢ Iff (Matrix.diagonal d).PosDef (∀ (i : n), LT.lt 0 (d i))
  -/
  refine ⟨fun h i => ?_, .diagonal⟩
  /-
    n : Type u_2
    R : Type u_3
    inst✝⁷ : Fintype n
    inst✝⁶ : CommRing R
    inst✝⁵ : PartialOrder R
    inst✝⁴ : StarRing R
    inst✝³ : StarOrderedRing R
    inst✝² : DecidableEq n
    inst✝¹ : NoZeroDivisors R
    inst✝ : Nontrivial R
    d : n → R
    h : (Matrix.diagonal d).PosDef
    i : n
    ⊢ LT.lt 0 (d i)
  -/
  have := h.2 (Pi.single i 1)
  simp only [mulVec_single, mul_one, dotProduct_diagonal', Pi.star_apply, Pi.single_eq_same,
    star_one, one_mul, Function.ne_iff, Pi.zero_apply] at this
  /-
    n : Type u_2
    R : Type u_3
    inst✝⁷ : Fintype n
    inst✝⁶ : CommRing R
    inst✝⁵ : PartialOrder R
    inst✝⁴ : StarRing R
    inst✝³ : StarOrderedRing R
    inst✝² : DecidableEq n
    inst✝¹ : NoZeroDivisors R
    inst✝ : Nontrivial R
    d : n → R
    h : (Matrix.diagonal d).PosDef
    i : n
    this : (Exists fun a => Ne (Pi.single i 1 a) 0) → LT.lt 0 (d i)
    ⊢ LT.lt 0 (d i)
  -/
  exact this ⟨i, by simp⟩
  /-
    🎉 no goals
  -/


protected theorem one [StarOrderedRing R] [DecidableEq n] [NoZeroDivisors R] :
    PosDef (1 : Matrix n n R) :=
                                   /-
                                     n : Type u_2
                                     R : Type u_3
                                     inst✝⁶ : Fintype n
                                     inst✝⁵ : CommRing R
                                     inst✝⁴ : PartialOrder R
                                     inst✝³ : StarRing R
                                     inst✝² : StarOrderedRing R
                                     inst✝¹ : DecidableEq n
                                     inst✝ : NoZeroDivisors R
                                     x : n → R
                                     hx : Ne x 0
                                     ⊢ LT.lt 0 (dotProduct (Star.star x) (Matrix.mulVec 1 x))
                                   -/
  ⟨isHermitian_one, fun x hx => by simpa only [one_mulVec, dotProduct_star_self_pos_iff]⟩
                                   /-
                                     🎉 no goals
                                   -/


protected theorem natCast [StarOrderedRing R] [DecidableEq n] [NoZeroDivisors R]
    (d : ℕ) (hd : d ≠ 0) :
    PosDef (d : Matrix n n R) :=
  ⟨isHermitian_natCast _, fun x hx => by
    /-
      n : Type u_2
      R : Type u_3
      inst✝⁶ : Fintype n
      inst✝⁵ : CommRing R
      inst✝⁴ : PartialOrder R
      inst✝³ : StarRing R
      inst✝² : StarOrderedRing R
      inst✝¹ : DecidableEq n
      inst✝ : NoZeroDivisors R
      d : Nat
      hd : Ne d 0
      x : n → R
      hx : Ne x 0
      ⊢ LT.lt 0 (dotProduct (Star.star x) ((↑d).mulVec x))
    -/
    simp only [natCast_mulVec, dotProduct_smul]
    /-
      n : Type u_2
      R : Type u_3
      inst✝⁶ : Fintype n
      inst✝⁵ : CommRing R
      inst✝⁴ : PartialOrder R
      inst✝³ : StarRing R
      inst✝² : StarOrderedRing R
      inst✝¹ : DecidableEq n
      inst✝ : NoZeroDivisors R
      d : Nat
      hd : Ne d 0
      x : n → R
      hx : Ne x 0
      ⊢ LT.lt 0 (HSMul.hSMul (↑d) (dotProduct (Star.star x) x))
    -/
    rw [Nat.cast_smul_eq_nsmul]
    /-
      n : Type u_2
      R : Type u_3
      inst✝⁶ : Fintype n
      inst✝⁵ : CommRing R
      inst✝⁴ : PartialOrder R
      inst✝³ : StarRing R
      inst✝² : StarOrderedRing R
      inst✝¹ : DecidableEq n
      inst✝ : NoZeroDivisors R
      d : Nat
      hd : Ne d 0
      x : n → R
      hx : Ne x 0
      ⊢ LT.lt 0 (HSMul.hSMul d (dotProduct (Star.star x) x))
    -/
    exact nsmul_pos (dotProduct_star_self_pos_iff.mpr hx) hd⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem _root_.Matrix.posDef_natCast_iff [StarOrderedRing R] [DecidableEq n] [NoZeroDivisors R]
    [Nonempty n] [Nontrivial R] {d : ℕ} :
    PosDef (d : Matrix n n R) ↔ 0 < d :=
                                  /-
                                    n : Type u_2
                                    R : Type u_3
                                    inst✝⁸ : Fintype n
                                    inst✝⁷ : CommRing R
                                    inst✝⁶ : PartialOrder R
                                    inst✝⁵ : StarRing R
                                    inst✝⁴ : StarOrderedRing R
                                    inst✝³ : DecidableEq n
                                    inst✝² : NoZeroDivisors R
                                    inst✝¹ : Nonempty n
                                    inst✝ : Nontrivial R
                                    d : Nat
                                    ⊢ Iff (n → LT.lt 0 ↑d) (LT.lt 0 d)
                                  -/
  posDef_diagonal_iff.trans <| by simp
                                  /-
                                    🎉 no goals
                                  -/

-- See note [no_index around OfNat.ofNat]

protected theorem ofNat [StarOrderedRing R] [DecidableEq n] [NoZeroDivisors R]
    (d : ℕ) [d.AtLeastTwo] :
    PosDef (no_index (OfNat.ofNat d) : Matrix n n R) :=
  .natCast d (NeZero.ne _)


protected theorem intCast [StarOrderedRing R] [DecidableEq n] [NoZeroDivisors R]
    (d : ℤ) (hd : 0 < d) :
    PosDef (d : Matrix n n R) :=
  ⟨isHermitian_intCast _, fun x hx => by
    /-
      n : Type u_2
      R : Type u_3
      inst✝⁶ : Fintype n
      inst✝⁵ : CommRing R
      inst✝⁴ : PartialOrder R
      inst✝³ : StarRing R
      inst✝² : StarOrderedRing R
      inst✝¹ : DecidableEq n
      inst✝ : NoZeroDivisors R
      d : Int
      hd : LT.lt 0 d
      x : n → R
      hx : Ne x 0
      ⊢ LT.lt 0 (dotProduct (Star.star x) ((↑d).mulVec x))
    -/
    simp only [intCast_mulVec, dotProduct_smul]
    /-
      n : Type u_2
      R : Type u_3
      inst✝⁶ : Fintype n
      inst✝⁵ : CommRing R
      inst✝⁴ : PartialOrder R
      inst✝³ : StarRing R
      inst✝² : StarOrderedRing R
      inst✝¹ : DecidableEq n
      inst✝ : NoZeroDivisors R
      d : Int
      hd : LT.lt 0 d
      x : n → R
      hx : Ne x 0
      ⊢ LT.lt 0 (HSMul.hSMul (↑d) (dotProduct (Star.star x) x))
    -/
    rw [Int.cast_smul_eq_zsmul]
    /-
      n : Type u_2
      R : Type u_3
      inst✝⁶ : Fintype n
      inst✝⁵ : CommRing R
      inst✝⁴ : PartialOrder R
      inst✝³ : StarRing R
      inst✝² : StarOrderedRing R
      inst✝¹ : DecidableEq n
      inst✝ : NoZeroDivisors R
      d : Int
      hd : LT.lt 0 d
      x : n → R
      hx : Ne x 0
      ⊢ LT.lt 0 (HSMul.hSMul d (dotProduct (Star.star x) x))
    -/
    exact zsmul_pos (dotProduct_star_self_pos_iff.mpr hx) hd⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem _root_.Matrix.posDef_intCast_iff [StarOrderedRing R] [DecidableEq n] [NoZeroDivisors R]
    [Nonempty n] [Nontrivial R] {d : ℤ} :
    PosDef (d : Matrix n n R) ↔ 0 < d :=
                                  /-
                                    n : Type u_2
                                    R : Type u_3
                                    inst✝⁸ : Fintype n
                                    inst✝⁷ : CommRing R
                                    inst✝⁶ : PartialOrder R
                                    inst✝⁵ : StarRing R
                                    inst✝⁴ : StarOrderedRing R
                                    inst✝³ : DecidableEq n
                                    inst✝² : NoZeroDivisors R
                                    inst✝¹ : Nonempty n
                                    inst✝ : Nontrivial R
                                    d : Int
                                    ⊢ Iff (n → LT.lt 0 ↑d) (LT.lt 0 d)
                                  -/
  posDef_diagonal_iff.trans <| by simp
                                  /-
                                    🎉 no goals
                                  -/


protected lemma add_posSemidef [AddLeftMono R]
    {A : Matrix m m R} {B : Matrix m m R}
    (hA : A.PosDef) (hB : B.PosSemidef) : (A + B).PosDef :=
  ⟨hA.isHermitian.add hB.isHermitian, fun x hx => by
    /-
      m : Type u_1
      R : Type u_3
      inst✝⁴ : Fintype m
      inst✝³ : CommRing R
      inst✝² : PartialOrder R
      inst✝¹ : StarRing R
      inst✝ : AddLeftMono R
      A B : Matrix m m R
      hA : A.PosDef
      hB : B.PosSemidef
      x : m → R
      hx : Ne x 0
      ⊢ LT.lt 0 (dotProduct (Star.star x) ((HAdd.hAdd A B).mulVec x))
    -/
    rw [add_mulVec, dotProduct_add]
    /-
      m : Type u_1
      R : Type u_3
      inst✝⁴ : Fintype m
      inst✝³ : CommRing R
      inst✝² : PartialOrder R
      inst✝¹ : StarRing R
      inst✝ : AddLeftMono R
      A B : Matrix m m R
      hA : A.PosDef
      hB : B.PosSemidef
      x : m → R
      hx : Ne x 0
      ⊢ LT.lt 0 (HAdd.hAdd (dotProduct (Star.star x) (A.mulVec x)) (dotProduct (Star …
    -/
    exact add_pos_of_pos_of_nonneg (hA.2 x hx) (hB.2 x)⟩
    /-
      🎉 no goals
    -/


protected lemma posSemidef_add [AddLeftMono R]
    {A : Matrix m m R} {B : Matrix m m R}
    (hA : A.PosSemidef) (hB : B.PosDef) : (A + B).PosDef :=
  ⟨hA.isHermitian.add hB.isHermitian, fun x hx => by
    /-
      m : Type u_1
      R : Type u_3
      inst✝⁴ : Fintype m
      inst✝³ : CommRing R
      inst✝² : PartialOrder R
      inst✝¹ : StarRing R
      inst✝ : AddLeftMono R
      A B : Matrix m m R
      hA : A.PosSemidef
      hB : B.PosDef
      x : m → R
      hx : Ne x 0
      ⊢ LT.lt 0 (dotProduct (Star.star x) ((HAdd.hAdd A B).mulVec x))
    -/
    rw [add_mulVec, dotProduct_add]
    /-
      m : Type u_1
      R : Type u_3
      inst✝⁴ : Fintype m
      inst✝³ : CommRing R
      inst✝² : PartialOrder R
      inst✝¹ : StarRing R
      inst✝ : AddLeftMono R
      A B : Matrix m m R
      hA : A.PosSemidef
      hB : B.PosDef
      x : m → R
      hx : Ne x 0
      ⊢ LT.lt 0 (HAdd.hAdd (dotProduct (Star.star x) (A.mulVec x)) (dotProduct (Star …
    -/
    exact add_pos_of_nonneg_of_pos (hA.2 x) (hB.2 x hx)⟩
    /-
      🎉 no goals
    -/


protected lemma add [AddLeftMono R] {A : Matrix m m R} {B : Matrix m m R}
    (hA : A.PosDef) (hB : B.PosDef) : (A + B).PosDef :=
  hA.add_posSemidef hB.posSemidef


theorem of_toQuadraticForm' [DecidableEq n] {M : Matrix n n ℝ} (hM : M.IsSymm)
    (hMq : M.toQuadraticMap'.PosDef) : M.PosDef := by
  /-
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    M : Matrix n n Real
    hM : M.IsSymm
    hMq : M.toQuadraticMap'.PosDef
    ⊢ M.PosDef
  -/
  refine ⟨hM, fun x hx => ?_⟩
  simp only [toQuadraticMap', QuadraticMap.PosDef, LinearMap.BilinMap.toQuadraticMap_apply,
    toLinearMap₂'_apply'] at hMq
  /-
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    M : Matrix n n Real
    hM : M.IsSymm
    x : n → Real
    hx : Ne x 0
    hMq : ∀ (x : n → Real), Ne x 0 → LT.lt 0 (dotProduct x (M.mulVec x))
    ⊢ LT.lt 0 (dotProduct (Star.star x) (M.mulVec x))
  -/
  apply hMq x hx
  /-
    🎉 no goals
  -/


theorem toQuadraticForm' [DecidableEq n] {M : Matrix n n ℝ} (hM : M.PosDef) :
    M.toQuadraticMap'.PosDef := by
  /-
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    M : Matrix n n Real
    hM : M.PosDef
    ⊢ M.toQuadraticMap'.PosDef
  -/
  intro x hx
  simp only [Matrix.toQuadraticMap', LinearMap.BilinMap.toQuadraticMap_apply,
    toLinearMap₂'_apply']
  /-
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    M : Matrix n n Real
    hM : M.PosDef
    x : n → Real
    hx : Ne x 0
    ⊢ LT.lt 0 (dotProduct x (M.mulVec x))
  -/
  apply hM.2 x hx
  /-
    🎉 no goals
  -/


/-- The eigenvalues of a positive definite matrix are positive -/
lemma eigenvalues_pos [DecidableEq n] {A : Matrix n n 𝕜}
    (hA : Matrix.PosDef A) (i : n) : 0 < hA.1.eigenvalues i := by
  /-
    n : Type u_2
    𝕜 : Type u_4
    inst✝² : Fintype n
    inst✝¹ : RCLike 𝕜
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    hA : A.PosDef
    i : n
    ⊢ LT.lt 0 (⋯.eigenvalues i)
  -/
  simp only [hA.1.eigenvalues_eq]
  /-
    n : Type u_2
    𝕜 : Type u_4
    inst✝² : Fintype n
    inst✝¹ : RCLike 𝕜
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    hA : A.PosDef
    i : n
    ⊢ LT.lt 0 (RCLike.re (dotProduct (Star.star ((WithLp.equiv 2 (n → 𝕜)) (⋯.eigen …
  -/
  exact hA.re_dotProduct_pos <| hA.1.eigenvectorBasis.orthonormal.ne_zero i
  /-
    🎉 no goals
  -/


theorem det_pos [DecidableEq n] {M : Matrix n n 𝕜} (hM : M.PosDef) : 0 < det M := by
  /-
    n : Type u_2
    𝕜 : Type u_4
    inst✝² : Fintype n
    inst✝¹ : RCLike 𝕜
    inst✝ : DecidableEq n
    M : Matrix n n 𝕜
    hM : M.PosDef
    ⊢ LT.lt 0 M.det
  -/
  rw [hM.isHermitian.det_eq_prod_eigenvalues]
  /-
    n : Type u_2
    𝕜 : Type u_4
    inst✝² : Fintype n
    inst✝¹ : RCLike 𝕜
    inst✝ : DecidableEq n
    M : Matrix n n 𝕜
    hM : M.PosDef
    ⊢ LT.lt 0 (Finset.univ.prod fun i => ↑(⋯.eigenvalues i))
  -/
  apply Finset.prod_pos
  /-
    case h0
    n : Type u_2
    𝕜 : Type u_4
    inst✝² : Fintype n
    inst✝¹ : RCLike 𝕜
    inst✝ : DecidableEq n
    M : Matrix n n 𝕜
    hM : M.PosDef
    ⊢ ∀ (i : n), Membership.mem Finset.univ i → LT.lt 0 ↑(⋯.eigenvalues i)
  -/
  intro i _
  /-
    case h0
    n : Type u_2
    𝕜 : Type u_4
    inst✝² : Fintype n
    inst✝¹ : RCLike 𝕜
    inst✝ : DecidableEq n
    M : Matrix n n 𝕜
    hM : M.PosDef
    i : n
    a✝ : Membership.mem Finset.univ i
    ⊢ LT.lt 0 ↑(⋯.eigenvalues i)
  -/
  simpa using hM.eigenvalues_pos i
  /-
    🎉 no goals
  -/


theorem isUnit [DecidableEq n] {M : Matrix n n 𝕜} (hM : M.PosDef) : IsUnit M :=
  isUnit_iff_isUnit_det _ |>.2 <| hM.det_pos.ne'.isUnit


protected theorem inv [DecidableEq n] {M : Matrix n n 𝕜} (hM : M.PosDef) : M⁻¹.PosDef := by
  /-
    n : Type u_2
    𝕜 : Type u_4
    inst✝² : Fintype n
    inst✝¹ : RCLike 𝕜
    inst✝ : DecidableEq n
    M : Matrix n n 𝕜
    hM : M.PosDef
    ⊢ (Inv.inv M).PosDef
  -/
  refine ⟨hM.isHermitian.inv, fun x hx => ?_⟩
  /-
    n : Type u_2
    𝕜 : Type u_4
    inst✝² : Fintype n
    inst✝¹ : RCLike 𝕜
    inst✝ : DecidableEq n
    M : Matrix n n 𝕜
    hM : M.PosDef
    x : n → 𝕜
    hx : Ne x 0
    ⊢ LT.lt 0 (dotProduct (Star.star x) ((Inv.inv M).mulVec x))
  -/
  have := hM.2 (M⁻¹ *ᵥ x) ((Matrix.mulVec_injective_iff_isUnit.mpr ?_ |>.ne_iff' ?_).2 hx)
    /-
      case refine_3
      n : Type u_2
      𝕜 : Type u_4
      inst✝² : Fintype n
      inst✝¹ : RCLike 𝕜
      inst✝ : DecidableEq n
      M : Matrix n n 𝕜
      hM : M.PosDef
      x : n → 𝕜
      hx : Ne x 0
      this : LT.lt 0 (dotProduct (Star.star ((Inv.inv M).mulVec x)) (M.mulVec ((Inv. …
      ⊢ LT.lt 0 (dotProduct (Star.star x) ((Inv.inv M).mulVec x))
    -/
  · let _inst := hM.isUnit.invertible
    rwa [star_mulVec, mulVec_mulVec, Matrix.mul_inv_of_invertible, one_mulVec,
      ← star_pos_iff, ← star_mulVec, ← star_dotProduct] at this
    /-
      case refine_1
      n : Type u_2
      𝕜 : Type u_4
      inst✝² : Fintype n
      inst✝¹ : RCLike 𝕜
      inst✝ : DecidableEq n
      M : Matrix n n 𝕜
      hM : M.PosDef
      x : n → 𝕜
      hx : Ne x 0
      ⊢ IsUnit (Inv.inv M)
    -/
  · simpa using hM.isUnit
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      n : Type u_2
      𝕜 : Type u_4
      inst✝² : Fintype n
      inst✝¹ : RCLike 𝕜
      inst✝ : DecidableEq n
      M : Matrix n n 𝕜
      hM : M.PosDef
      x : n → 𝕜
      hx : Ne x 0
      ⊢ Eq ((Inv.inv M).mulVec 0) 0
    -/
  · simp
    /-
      🎉 no goals
    -/


@[simp]
theorem _root_.Matrix.posDef_inv_iff [DecidableEq n] {M : Matrix n n 𝕜} :
    M⁻¹.PosDef ↔ M.PosDef :=
  ⟨fun h =>
    letI := (Matrix.isUnit_nonsing_inv_iff.1 <| h.isUnit).invertible
    Matrix.inv_inv_of_invertible M ▸ h.inv, (·.inv)⟩


theorem posDef_of_toMatrix' [DecidableEq n] {Q : QuadraticForm ℝ (n → ℝ)}
    (hQ : Q.toMatrix'.PosDef) : Q.PosDef := by
  rw [← toQuadraticMap_associated ℝ Q,
    ← (LinearMap.toMatrix₂' ℝ).left_inv ((associatedHom (R := ℝ) ℝ) Q)]
  /-
    n : Type u_1
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    Q : QuadraticForm Real (n → Real)
    hQ : (QuadraticMap.toMatrix' Q).PosDef
    ⊢ (LinearMap.BilinMap.toQuadraticMap ((LinearMap.toMatrix₂' Real).invFun ((↑(L …
  -/
  exact hQ.toQuadraticForm'
  /-
    🎉 no goals
  -/


theorem posDef_toMatrix' [DecidableEq n] {Q : QuadraticForm ℝ (n → ℝ)} (hQ : Q.PosDef) :
    Q.toMatrix'.PosDef := by
  rw [← toQuadraticMap_associated ℝ Q, ←
    (LinearMap.toMatrix₂' ℝ).left_inv ((associatedHom (R := ℝ) ℝ) Q)] at hQ
  /-
    n : Type u_1
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    Q : QuadraticForm Real (n → Real)
    hQ : (LinearMap.BilinMap.toQuadraticMap ((LinearMap.toMatrix₂' Real).invFun (( …
    ⊢ (QuadraticMap.toMatrix' Q).PosDef
  -/
  exact .of_toQuadraticForm' (isSymm_toMatrix' Q) hQ
  /-
    🎉 no goals
  -/


/-- A positive definite matrix `M` induces a norm `‖x‖ = sqrt (re xᴴMx)`. -/
noncomputable abbrev NormedAddCommGroup.ofMatrix {M : Matrix n n 𝕜} (hM : M.PosDef) :
    NormedAddCommGroup (n → 𝕜) :=
  @InnerProductSpace.Core.toNormedAddCommGroup _ _ _ _ _
    { inner := fun x y => dotProduct (star x) (M *ᵥ y)
      conj_symm := fun x y => by
        /-
          𝕜 : Type u_1
          inst✝¹ : RCLike 𝕜
          n : Type u_2
          inst✝ : Fintype n
          M : Matrix n n 𝕜
          hM : M.PosDef
          x y : n → 𝕜
          ⊢ Eq ((starRingEnd 𝕜) (Inner.inner y x)) (Inner.inner x y)
        -/
        dsimp only [Inner.inner]
        rw [star_dotProduct, starRingEnd_apply, star_star, star_mulVec, dotProduct_mulVec,
          hM.isHermitian.eq]
      nonneg_re := fun x => by
        /-
          𝕜 : Type u_1
          inst✝¹ : RCLike 𝕜
          n : Type u_2
          inst✝ : Fintype n
          M : Matrix n n 𝕜
          hM : M.PosDef
          x : n → 𝕜
          ⊢ LE.le 0 (RCLike.re (Inner.inner x x))
        -/
        by_cases h : x = 0
          /-
            case pos
            𝕜 : Type u_1
            inst✝¹ : RCLike 𝕜
            n : Type u_2
            inst✝ : Fintype n
            M : Matrix n n 𝕜
            hM : M.PosDef
            x : n → 𝕜
            h : Eq x 0
            ⊢ LE.le 0 (RCLike.re (Inner.inner x x))
          -/
        · simp [h]
          /-
            🎉 no goals
          -/
          /-
            case neg
            𝕜 : Type u_1
            inst✝¹ : RCLike 𝕜
            n : Type u_2
            inst✝ : Fintype n
            M : Matrix n n 𝕜
            hM : M.PosDef
            x : n → 𝕜
            h : Not (Eq x 0)
            ⊢ LE.le 0 (RCLike.re (Inner.inner x x))
          -/
        · exact le_of_lt (hM.re_dotProduct_pos h)
          /-
            🎉 no goals
          -/
      definite := fun x (hx : dotProduct _ _ = 0) => by
        /-
          𝕜 : Type u_1
          inst✝¹ : RCLike 𝕜
          n : Type u_2
          inst✝ : Fintype n
          M : Matrix n n 𝕜
          hM : M.PosDef
          x : n → 𝕜
          hx : Eq (dotProduct (Star.star x) (M.mulVec x)) 0
          ⊢ Eq x 0
        -/
        by_contra! h
                     /-
                       𝕜 : Type u_1
                       inst✝¹ : RCLike 𝕜
                       n : Type u_2
                       inst✝ : Fintype n
                       M : Matrix n n 𝕜
                       hM : M.PosDef
                       ⊢ ∀ (x y z : n → 𝕜), Eq (Inner.inner (HAdd.hAdd x y) z) (HAdd.hAdd (Inner.inne …
                     -/
        /-
          𝕜 : Type u_1
          inst✝¹ : RCLike 𝕜
          n : Type u_2
          inst✝ : Fintype n
          M : Matrix n n 𝕜
          hM : M.PosDef
          x : n → 𝕜
          hx : Eq (dotProduct (Star.star x) (M.mulVec x)) 0
          h : Ne x 0
          ⊢ False
        -/
                     /-
                       🎉 no goals
                     -/
        simpa [hx, lt_irrefl] using hM.re_dotProduct_pos h
        /-
          𝕜 : Type u_1
          inst✝¹ : RCLike 𝕜
          n : Type u_2
          inst✝ : Fintype n
          M : Matrix n n 𝕜
          hM : M.PosDef
          x y : n → 𝕜
          r : 𝕜
          ⊢ Eq (Inner.inner (HSMul.hSMul r x) y) (HMul.hMul ((starRingEnd 𝕜) r) (Inner.i …
        -/
        /-
          🎉 no goals
        -/
        /-
          𝕜 : Type u_1
          inst✝¹ : RCLike 𝕜
          n : Type u_2
          inst✝ : Fintype n
          M : Matrix n n 𝕜
          hM : M.PosDef
          x y : n → 𝕜
          r : 𝕜
          ⊢ Eq (dotProduct (Star.star (HSMul.hSMul r x)) (M.mulVec y)) (HMul.hMul ((star …
        -/
      add_left := by simp only [star_add, add_dotProduct, eq_self_iff_true, forall_const]
        /-
          🎉 no goals
        -/
      smul_left := fun x y r => by
        simp only
        rw [← smul_eq_mul, ← smul_dotProduct, starRingEnd_apply, ← star_smul] }


/-- A positive definite matrix `M` induces an inner product `⟪x, y⟫ = xᴴMy`. -/
def InnerProductSpace.ofMatrix {M : Matrix n n 𝕜} (hM : M.PosDef) :
    @InnerProductSpace 𝕜 (n → 𝕜) _ (NormedAddCommGroup.ofMatrix hM).toSeminormedAddCommGroup :=
  InnerProductSpace.ofCore _


