/-- The transvection matrix `transvection i j c` is equal to the identity plus `c` at position
`(i, j)`. Multiplying by it on the left (as in `transvection i j c * M`) corresponds to adding
`c` times the `j`-th row of `M` to its `i`-th row. Multiplying by it on the right corresponds
to adding `c` times the `i`-th column to the `j`-th column. -/
def transvection (c : R) : Matrix n n R :=
  1 + Matrix.stdBasisMatrix i j c


@[simp]
                                                               /-
                                                                 n : Type u_1
                                                                 R : Type u₂
                                                                 inst✝¹ : DecidableEq n
                                                                 inst✝ : CommRing R
                                                                 i j : n
                                                                 ⊢ Eq (Matrix.transvection i j 0) 1
                                                               -/
theorem transvection_zero : transvection i j (0 : R) = 1 := by simp [transvection]
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- A transvection matrix is obtained from the identity by adding `c` times the `j`-th row to
the `i`-th row. -/
theorem updateRow_eq_transvection [Finite n] (c : R) :
    updateRow (1 : Matrix n n R) i ((1 : Matrix n n R) i + c • (1 : Matrix n n R) j) =
      transvection i j c := by
  /-
    n : Type u_1
    R : Type u₂
    inst✝² : DecidableEq n
    inst✝¹ : CommRing R
    i j : n
    inst✝ : Finite n
    c : R
    ⊢ Eq (Matrix.updateRow 1 i (HAdd.hAdd (1 i) (HSMul.hSMul c (1 j)))) (Matrix.tr …
  -/
  cases nonempty_fintype n
  /-
    case intro
    n : Type u_1
    R : Type u₂
    inst✝² : DecidableEq n
    inst✝¹ : CommRing R
    i j : n
    inst✝ : Finite n
    c : R
    val✝ : Fintype n
    ⊢ Eq (Matrix.updateRow 1 i (HAdd.hAdd (1 i) (HSMul.hSMul c (1 j)))) (Matrix.tr …
  -/
  ext a b
  /-
    case intro.a
    n : Type u_1
    R : Type u₂
    inst✝² : DecidableEq n
    inst✝¹ : CommRing R
    i j : n
    inst✝ : Finite n
    c : R
    val✝ : Fintype n
    a b : n
    ⊢ Eq (Matrix.updateRow 1 i (HAdd.hAdd (1 i) (HSMul.hSMul c (1 j))) a b) (Matri …
  -/
  by_cases ha : i = a
    /-
      case pos
      n : Type u_1
      R : Type u₂
      inst✝² : DecidableEq n
      inst✝¹ : CommRing R
      i j : n
      inst✝ : Finite n
      c : R
      val✝ : Fintype n
      a b : n
      ha : Eq i a
      ⊢ Eq (Matrix.updateRow 1 i (HAdd.hAdd (1 i) (HSMul.hSMul c (1 j))) a b) (Matri …
    -/
  · by_cases hb : j = b
    · simp only [ha, updateRow_self, Pi.add_apply, one_apply, Pi.smul_apply, hb, ↓reduceIte,
        smul_eq_mul, mul_one, transvection, add_apply, StdBasisMatrix.apply_same]
    · simp only [ha, updateRow_self, Pi.add_apply, one_apply, Pi.smul_apply, hb, ↓reduceIte,
        smul_eq_mul, mul_zero, add_zero, transvection, add_apply, and_false, not_false_eq_true,
        StdBasisMatrix.apply_of_ne]
  · simp only [updateRow_ne, transvection, ha, Ne.symm ha, StdBasisMatrix.apply_of_ne, add_zero,
      Algebra.id.smul_eq_mul, Ne, not_false_iff, DMatrix.add_apply, Pi.smul_apply,
      mul_zero, false_and, add_apply]


theorem transvection_mul_transvection_same (h : i ≠ j) (c d : R) :
    transvection i j c * transvection i j d = transvection i j (c + d) := by
  simp [transvection, Matrix.add_mul, Matrix.mul_add, h, h.symm, add_smul, add_assoc,
    stdBasisMatrix_add]


@[simp]
theorem transvection_mul_apply_same (b : n) (c : R) (M : Matrix n n R) :
                                                           /-
                                                             n : Type u_1
                                                             R : Type u₂
                                                             inst✝² : DecidableEq n
                                                             inst✝¹ : CommRing R
                                                             i j : n
                                                             inst✝ : Fintype n
                                                             b : n
                                                             c : R
                                                             M : Matrix n n R
                                                             ⊢ Eq (HMul.hMul (Matrix.transvection i j c) M i b) (HAdd.hAdd (M i b) (HMul.hM …
                                                           -/
    (transvection i j c * M) i b = M i b + c * M j b := by simp [transvection, Matrix.add_mul]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp]
theorem mul_transvection_apply_same (a : n) (c : R) (M : Matrix n n R) :
    (M * transvection i j c) a j = M a j + c * M a i := by
  /-
    n : Type u_1
    R : Type u₂
    inst✝² : DecidableEq n
    inst✝¹ : CommRing R
    i j : n
    inst✝ : Fintype n
    a : n
    c : R
    M : Matrix n n R
    ⊢ Eq (HMul.hMul M (Matrix.transvection i j c) a j) (HAdd.hAdd (M a j) (HMul.hM …
  -/
  simp [transvection, Matrix.mul_add, mul_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem transvection_mul_apply_of_ne (a b : n) (ha : a ≠ i) (c : R) (M : Matrix n n R) :
                                               /-
                                                 n : Type u_1
                                                 R : Type u₂
                                                 inst✝² : DecidableEq n
                                                 inst✝¹ : CommRing R
                                                 i j : n
                                                 inst✝ : Fintype n
                                                 a b : n
                                                 ha : Ne a i
                                                 c : R
                                                 M : Matrix n n R
                                                 ⊢ Eq (HMul.hMul (Matrix.transvection i j c) M a b) (M a b)
                                               -/
    (transvection i j c * M) a b = M a b := by simp [transvection, Matrix.add_mul, ha]
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
theorem mul_transvection_apply_of_ne (a b : n) (hb : b ≠ j) (c : R) (M : Matrix n n R) :
                                               /-
                                                 n : Type u_1
                                                 R : Type u₂
                                                 inst✝² : DecidableEq n
                                                 inst✝¹ : CommRing R
                                                 i j : n
                                                 inst✝ : Fintype n
                                                 a b : n
                                                 hb : Ne b j
                                                 c : R
                                                 M : Matrix n n R
                                                 ⊢ Eq (HMul.hMul M (Matrix.transvection i j c) a b) (M a b)
                                               -/
    (M * transvection i j c) a b = M a b := by simp [transvection, Matrix.mul_add, hb]
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
theorem det_transvection_of_ne (h : i ≠ j) (c : R) : det (transvection i j c) = 1 := by
  /-
    n : Type u_1
    R : Type u₂
    inst✝² : DecidableEq n
    inst✝¹ : CommRing R
    i j : n
    inst✝ : Fintype n
    h : Ne i j
    c : R
    ⊢ Eq (Matrix.transvection i j c).det 1
  -/
  rw [← updateRow_eq_transvection i j, det_updateRow_add_smul_self _ h, det_one]
  /-
    🎉 no goals
  -/


/-- A structure containing all the information from which one can build a nontrivial transvection.
This structure is easier to manipulate than transvections as one has a direct access to all the
relevant fields. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed @[nolint has_nonempty_instance]
structure TransvectionStruct where
  (i j : n)
  hij : i ≠ j
  c : R


instance [Nontrivial n] : Nonempty (TransvectionStruct n R) := by
  /-
    n : Type u_1
    p : Type u_2
    R : Type u₂
    𝕜 : Type u_3
    inst✝⁴ : Field 𝕜
    inst✝³ : DecidableEq n
    inst✝² : DecidableEq p
    inst✝¹ : CommRing R
    i j : n
    inst✝ : Nontrivial n
    ⊢ Nonempty (Matrix.TransvectionStruct n R)
  -/
  choose x y hxy using exists_pair_ne n
  /-
    n : Type u_1
    p : Type u_2
    R : Type u₂
    𝕜 : Type u_3
    inst✝⁴ : Field 𝕜
    inst✝³ : DecidableEq n
    inst✝² : DecidableEq p
    inst✝¹ : CommRing R
    i j : n
    inst✝ : Nontrivial n
    x y : n
    hxy : Ne x y
    ⊢ Nonempty (Matrix.TransvectionStruct n R)
  -/
  exact ⟨⟨x, y, hxy, 0⟩⟩
  /-
    🎉 no goals
  -/


/-- Associating to a `transvection_struct` the corresponding transvection matrix. -/
def toMatrix (t : TransvectionStruct n R) : Matrix n n R :=
  transvection t.i t.j t.c


@[simp]
theorem toMatrix_mk (i j : n) (hij : i ≠ j) (c : R) :
    TransvectionStruct.toMatrix ⟨i, j, hij, c⟩ = transvection i j c :=
  rfl


@[simp]
protected theorem det [Fintype n] (t : TransvectionStruct n R) : det t.toMatrix = 1 :=
  det_transvection_of_ne _ _ t.hij _


@[simp]
theorem det_toMatrix_prod [Fintype n] (L : List (TransvectionStruct n 𝕜)) :
    det (L.map toMatrix).prod = 1 := by
  /-
    n : Type u_1
    𝕜 : Type u_3
    inst✝² : Field 𝕜
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    L : List (Matrix.TransvectionStruct n 𝕜)
    ⊢ Eq (List.map Matrix.TransvectionStruct.toMatrix L).prod.det 1
  -/
  induction' L with t L IH
    /-
      case nil
      n : Type u_1
      𝕜 : Type u_3
      inst✝² : Field 𝕜
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      ⊢ Eq (List.map Matrix.TransvectionStruct.toMatrix List.nil).prod.det 1
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      n : Type u_1
      𝕜 : Type u_3
      inst✝² : Field 𝕜
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      t : Matrix.TransvectionStruct n 𝕜
      L : List (Matrix.TransvectionStruct n 𝕜)
      IH : Eq (List.map Matrix.TransvectionStruct.toMatrix L).prod.det 1
      ⊢ Eq (List.map Matrix.TransvectionStruct.toMatrix (List.cons t L)).prod.det 1
    -/
  · simp [IH]
    /-
      🎉 no goals
    -/


/-- The inverse of a `TransvectionStruct`, designed so that `t.inv.toMatrix` is the inverse of
`t.toMatrix`. -/
@[simps]
protected def inv (t : TransvectionStruct n R) : TransvectionStruct n R where
  i := t.i
  j := t.j
  hij := t.hij
  c := -t.c


theorem inv_mul (t : TransvectionStruct n R) : t.inv.toMatrix * t.toMatrix = 1 := by
  /-
    n : Type u_1
    R : Type u₂
    inst✝² : DecidableEq n
    inst✝¹ : CommRing R
    inst✝ : Fintype n
    t : Matrix.TransvectionStruct n R
    ⊢ Eq (HMul.hMul t.inv.toMatrix t.toMatrix) 1
  -/
  rcases t with ⟨_, _, t_hij⟩
  /-
    case mk
    n : Type u_1
    R : Type u₂
    inst✝² : DecidableEq n
    inst✝¹ : CommRing R
    inst✝ : Fintype n
    i✝ j✝ : n
    t_hij : Ne i✝ j✝
    c✝ : R
    ⊢ Eq (HMul.hMul { i := i✝, j := j✝, hij := t_hij, c := c✝ }.inv.toMatrix { i : …
  -/
  simp [toMatrix, transvection_mul_transvection_same, t_hij]
  /-
    🎉 no goals
  -/


theorem mul_inv (t : TransvectionStruct n R) : t.toMatrix * t.inv.toMatrix = 1 := by
  /-
    n : Type u_1
    R : Type u₂
    inst✝² : DecidableEq n
    inst✝¹ : CommRing R
    inst✝ : Fintype n
    t : Matrix.TransvectionStruct n R
    ⊢ Eq (HMul.hMul t.toMatrix t.inv.toMatrix) 1
  -/
  rcases t with ⟨_, _, t_hij⟩
  /-
    case mk
    n : Type u_1
    R : Type u₂
    inst✝² : DecidableEq n
    inst✝¹ : CommRing R
    inst✝ : Fintype n
    i✝ j✝ : n
    t_hij : Ne i✝ j✝
    c✝ : R
    ⊢ Eq (HMul.hMul { i := i✝, j := j✝, hij := t_hij, c := c✝ }.toMatrix { i := i✝ …
  -/
  simp [toMatrix, transvection_mul_transvection_same, t_hij]
  /-
    🎉 no goals
  -/


theorem reverse_inv_prod_mul_prod (L : List (TransvectionStruct n R)) :
    (L.reverse.map (toMatrix ∘ TransvectionStruct.inv)).prod * (L.map toMatrix).prod = 1 := by
  /-
    n : Type u_1
    R : Type u₂
    inst✝² : DecidableEq n
    inst✝¹ : CommRing R
    inst✝ : Fintype n
    L : List (Matrix.TransvectionStruct n R)
    ⊢ Eq (HMul.hMul (List.map (Function.comp Matrix.TransvectionStruct.toMatrix Ma …
  -/
  induction' L with t L IH
    /-
      case nil
      n : Type u_1
      R : Type u₂
      inst✝² : DecidableEq n
      inst✝¹ : CommRing R
      inst✝ : Fintype n
      ⊢ Eq (HMul.hMul (List.map (Function.comp Matrix.TransvectionStruct.toMatrix Ma …
    -/
  · simp
    /-
      🎉 no goals
    -/
  · suffices
      (L.reverse.map (toMatrix ∘ TransvectionStruct.inv)).prod * (t.inv.toMatrix * t.toMatrix) *
          (L.map toMatrix).prod = 1
      by simpa [Matrix.mul_assoc]
    /-
      case cons
      n : Type u_1
      R : Type u₂
      inst✝² : DecidableEq n
      inst✝¹ : CommRing R
      inst✝ : Fintype n
      t : Matrix.TransvectionStruct n R
      L : List (Matrix.TransvectionStruct n R)
      IH : Eq (HMul.hMul (List.map (Function.comp Matrix.TransvectionStruct.toMatrix …
      ⊢ Eq (HMul.hMul (HMul.hMul (List.map (Function.comp Matrix.TransvectionStruct. …
    -/
    simpa [inv_mul] using IH
    /-
      🎉 no goals
    -/


theorem prod_mul_reverse_inv_prod (L : List (TransvectionStruct n R)) :
    (L.map toMatrix).prod * (L.reverse.map (toMatrix ∘ TransvectionStruct.inv)).prod = 1 := by
  /-
    n : Type u_1
    R : Type u₂
    inst✝² : DecidableEq n
    inst✝¹ : CommRing R
    inst✝ : Fintype n
    L : List (Matrix.TransvectionStruct n R)
    ⊢ Eq (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L).prod (List.map …
  -/
  induction' L with t L IH
    /-
      case nil
      n : Type u_1
      R : Type u₂
      inst✝² : DecidableEq n
      inst✝¹ : CommRing R
      inst✝ : Fintype n
      ⊢ Eq (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix List.nil).prod (L …
    -/
  · simp
    /-
      🎉 no goals
    -/
  · suffices
      t.toMatrix *
            ((L.map toMatrix).prod * (L.reverse.map (toMatrix ∘ TransvectionStruct.inv)).prod) *
          t.inv.toMatrix = 1
      by simpa [Matrix.mul_assoc]
    /-
      case cons
      n : Type u_1
      R : Type u₂
      inst✝² : DecidableEq n
      inst✝¹ : CommRing R
      inst✝ : Fintype n
      t : Matrix.TransvectionStruct n R
      L : List (Matrix.TransvectionStruct n R)
      IH : Eq (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L).prod (List. …
      ⊢ Eq (HMul.hMul (HMul.hMul t.toMatrix (HMul.hMul (List.map Matrix.Transvection …
    -/
    simp_rw [IH, Matrix.mul_one, t.mul_inv]
    /-
      🎉 no goals
    -/


/-- `M` is a scalar matrix if it commutes with every nontrivial transvection (elementary matrix). -/
theorem _root_.Matrix.mem_range_scalar_of_commute_transvectionStruct {M : Matrix n n R}
    (hM : ∀ t : TransvectionStruct n R, Commute t.toMatrix M) :
    M ∈ Set.range (Matrix.scalar n) := by
  /-
    n : Type u_1
    R : Type u₂
    inst✝² : DecidableEq n
    inst✝¹ : CommRing R
    inst✝ : Fintype n
    M : Matrix n n R
    hM : ∀ (t : Matrix.TransvectionStruct n R), Commute t.toMatrix M
    ⊢ Membership.mem (Set.range ⇑(Matrix.scalar n)) M
  -/
  refine mem_range_scalar_of_commute_stdBasisMatrix ?_
  /-
    n : Type u_1
    R : Type u₂
    inst✝² : DecidableEq n
    inst✝¹ : CommRing R
    inst✝ : Fintype n
    M : Matrix n n R
    hM : ∀ (t : Matrix.TransvectionStruct n R), Commute t.toMatrix M
    ⊢ Pairwise fun i j => Commute (Matrix.stdBasisMatrix i j 1) M
  -/
  intro i j hij
  /-
    n : Type u_1
    R : Type u₂
    inst✝² : DecidableEq n
    inst✝¹ : CommRing R
    inst✝ : Fintype n
    M : Matrix n n R
    hM : ∀ (t : Matrix.TransvectionStruct n R), Commute t.toMatrix M
    i j : n
    hij : Ne i j
    ⊢ Commute (Matrix.stdBasisMatrix i j 1) M
  -/
  simpa [transvection, mul_add, add_mul] using (hM ⟨i, j, hij, 1⟩).eq
  /-
    🎉 no goals
  -/


theorem _root_.Matrix.mem_range_scalar_iff_commute_transvectionStruct {M : Matrix n n R} :
    M ∈ Set.range (Matrix.scalar n) ↔ ∀ t : TransvectionStruct n R, Commute t.toMatrix M := by
  /-
    n : Type u_1
    R : Type u₂
    inst✝² : DecidableEq n
    inst✝¹ : CommRing R
    inst✝ : Fintype n
    M : Matrix n n R
    ⊢ Iff (Membership.mem (Set.range ⇑(Matrix.scalar n)) M) (∀ (t : Matrix.Transve …
  -/
  refine ⟨fun h t => ?_, mem_range_scalar_of_commute_transvectionStruct⟩
  /-
    n : Type u_1
    R : Type u₂
    inst✝² : DecidableEq n
    inst✝¹ : CommRing R
    inst✝ : Fintype n
    M : Matrix n n R
    h : Membership.mem (Set.range ⇑(Matrix.scalar n)) M
    t : Matrix.TransvectionStruct n R
    ⊢ Commute t.toMatrix M
  -/
  rw [mem_range_scalar_iff_commute_stdBasisMatrix] at h
  /-
    n : Type u_1
    R : Type u₂
    inst✝² : DecidableEq n
    inst✝¹ : CommRing R
    inst✝ : Fintype n
    M : Matrix n n R
    h : ∀ (i j : n), Ne i j → Commute (Matrix.stdBasisMatrix i j 1) M
    t : Matrix.TransvectionStruct n R
    ⊢ Commute t.toMatrix M
  -/
  refine (Commute.one_left M).add_left ?_
  /-
    n : Type u_1
    R : Type u₂
    inst✝² : DecidableEq n
    inst✝¹ : CommRing R
    inst✝ : Fintype n
    M : Matrix n n R
    h : ∀ (i j : n), Ne i j → Commute (Matrix.stdBasisMatrix i j 1) M
    t : Matrix.TransvectionStruct n R
    ⊢ Commute (Matrix.stdBasisMatrix t.i t.j t.c) M
  -/
  convert (h _ _ t.hij).smul_left t.c using 1
  /-
    case h.e'_3
    n : Type u_1
    R : Type u₂
    inst✝² : DecidableEq n
    inst✝¹ : CommRing R
    inst✝ : Fintype n
    M : Matrix n n R
    h : ∀ (i j : n), Ne i j → Commute (Matrix.stdBasisMatrix i j 1) M
    t : Matrix.TransvectionStruct n R
    ⊢ Eq (Matrix.stdBasisMatrix t.i t.j t.c) (HSMul.hSMul t.c (Matrix.stdBasisMatr …
  -/
  rw [smul_stdBasisMatrix, smul_eq_mul, mul_one]
  /-
    🎉 no goals
  -/


/-- Given a `TransvectionStruct` on `n`, define the corresponding `TransvectionStruct` on `n ⊕ p`
using the identity on `p`. -/
def sumInl (t : TransvectionStruct n R) : TransvectionStruct (n ⊕ p) R where
  i := inl t.i
  j := inl t.j
            /-
              n : Type u_1
              p : Type u_2
              R : Type u₂
              𝕜 : Type u_3
              inst✝³ : Field 𝕜
              inst✝² : DecidableEq n
              inst✝¹ : DecidableEq p
              inst✝ : CommRing R
              i j : n
              t : Matrix.TransvectionStruct n R
              ⊢ Ne (Sum.inl t.i) (Sum.inl t.j)
            -/
  hij := by simp [t.hij]
            /-
              🎉 no goals
            -/
  c := t.c


theorem toMatrix_sumInl (t : TransvectionStruct n R) :
    (t.sumInl p).toMatrix = fromBlocks t.toMatrix 0 0 1 := by
  /-
    n : Type u_1
    p : Type u_2
    R : Type u₂
    inst✝² : DecidableEq n
    inst✝¹ : DecidableEq p
    inst✝ : CommRing R
    t : Matrix.TransvectionStruct n R
    ⊢ Eq (Matrix.TransvectionStruct.sumInl p t).toMatrix (Matrix.fromBlocks t.toMa …
  -/
  cases t
  /-
    case mk
    n : Type u_1
    p : Type u_2
    R : Type u₂
    inst✝² : DecidableEq n
    inst✝¹ : DecidableEq p
    inst✝ : CommRing R
    i✝ j✝ : n
    hij✝ : Ne i✝ j✝
    c✝ : R
    ⊢ Eq (Matrix.TransvectionStruct.sumInl p { i := i✝, j := j✝, hij := hij✝, c := …
  -/
  ext a b
  /-
    case mk.a
    n : Type u_1
    p : Type u_2
    R : Type u₂
    inst✝² : DecidableEq n
    inst✝¹ : DecidableEq p
    inst✝ : CommRing R
    i✝ j✝ : n
    hij✝ : Ne i✝ j✝
    c✝ : R
    a b : Sum n p
    ⊢ Eq ((Matrix.TransvectionStruct.sumInl p { i := i✝, j := j✝, hij := hij✝, c : …
  -/
  cases' a with a a <;> cases' b with b b
    /-
      case mk.a.inl.inl
      n : Type u_1
      p : Type u_2
      R : Type u₂
      inst✝² : DecidableEq n
      inst✝¹ : DecidableEq p
      inst✝ : CommRing R
      i✝ j✝ : n
      hij✝ : Ne i✝ j✝
      c✝ : R
      a b : n
      ⊢ Eq ((Matrix.TransvectionStruct.sumInl p { i := i✝, j := j✝, hij := hij✝, c : …
    -/
                           /-
                             🎉 no goals
                           -/
  · by_cases h : a = b <;> simp [TransvectionStruct.sumInl, transvection, h, stdBasisMatrix]
                           /-
                             🎉 no goals
                           -/
    /-
      case mk.a.inl.inr
      n : Type u_1
      p : Type u_2
      R : Type u₂
      inst✝² : DecidableEq n
      inst✝¹ : DecidableEq p
      inst✝ : CommRing R
      i✝ j✝ : n
      hij✝ : Ne i✝ j✝
      c✝ : R
      a : n
      b : p
      ⊢ Eq ((Matrix.TransvectionStruct.sumInl p { i := i✝, j := j✝, hij := hij✝, c : …
    -/
  · simp [TransvectionStruct.sumInl, transvection]
    /-
      🎉 no goals
    -/
    /-
      case mk.a.inr.inl
      n : Type u_1
      p : Type u_2
      R : Type u₂
      inst✝² : DecidableEq n
      inst✝¹ : DecidableEq p
      inst✝ : CommRing R
      i✝ j✝ : n
      hij✝ : Ne i✝ j✝
      c✝ : R
      a : p
      b : n
      ⊢ Eq ((Matrix.TransvectionStruct.sumInl p { i := i✝, j := j✝, hij := hij✝, c : …
    -/
  · simp [TransvectionStruct.sumInl, transvection]
    /-
      🎉 no goals
    -/
    /-
      case mk.a.inr.inr
      n : Type u_1
      p : Type u_2
      R : Type u₂
      inst✝² : DecidableEq n
      inst✝¹ : DecidableEq p
      inst✝ : CommRing R
      i✝ j✝ : n
      hij✝ : Ne i✝ j✝
      c✝ : R
      a b : p
      ⊢ Eq ((Matrix.TransvectionStruct.sumInl p { i := i✝, j := j✝, hij := hij✝, c : …
    -/
                           /-
                             🎉 no goals
                           -/
  · by_cases h : a = b <;> simp [TransvectionStruct.sumInl, transvection, h]
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem sumInl_toMatrix_prod_mul [Fintype n] [Fintype p] (M : Matrix n n R)
    (L : List (TransvectionStruct n R)) (N : Matrix p p R) :
    (L.map (toMatrix ∘ sumInl p)).prod * fromBlocks M 0 0 N =
      fromBlocks ((L.map toMatrix).prod * M) 0 0 N := by
  /-
    n : Type u_1
    p : Type u_2
    R : Type u₂
    inst✝⁴ : DecidableEq n
    inst✝³ : DecidableEq p
    inst✝² : CommRing R
    inst✝¹ : Fintype n
    inst✝ : Fintype p
    M : Matrix n n R
    L : List (Matrix.TransvectionStruct n R)
    N : Matrix p p R
    ⊢ Eq (HMul.hMul (List.map (Function.comp Matrix.TransvectionStruct.toMatrix (M …
  -/
  induction' L with t L IH
    /-
      case nil
      n : Type u_1
      p : Type u_2
      R : Type u₂
      inst✝⁴ : DecidableEq n
      inst✝³ : DecidableEq p
      inst✝² : CommRing R
      inst✝¹ : Fintype n
      inst✝ : Fintype p
      M : Matrix n n R
      N : Matrix p p R
      ⊢ Eq (HMul.hMul (List.map (Function.comp Matrix.TransvectionStruct.toMatrix (M …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      n : Type u_1
      p : Type u_2
      R : Type u₂
      inst✝⁴ : DecidableEq n
      inst✝³ : DecidableEq p
      inst✝² : CommRing R
      inst✝¹ : Fintype n
      inst✝ : Fintype p
      M : Matrix n n R
      N : Matrix p p R
      t : Matrix.TransvectionStruct n R
      L : List (Matrix.TransvectionStruct n R)
      IH : Eq (HMul.hMul (List.map (Function.comp Matrix.TransvectionStruct.toMatrix …
      ⊢ Eq (HMul.hMul (List.map (Function.comp Matrix.TransvectionStruct.toMatrix (M …
    -/
  · simp [Matrix.mul_assoc, IH, toMatrix_sumInl, fromBlocks_multiply]
    /-
      🎉 no goals
    -/


@[simp]
theorem mul_sumInl_toMatrix_prod [Fintype n] [Fintype p] (M : Matrix n n R)
    (L : List (TransvectionStruct n R)) (N : Matrix p p R) :
    fromBlocks M 0 0 N * (L.map (toMatrix ∘ sumInl p)).prod =
      fromBlocks (M * (L.map toMatrix).prod) 0 0 N := by
  /-
    n : Type u_1
    p : Type u_2
    R : Type u₂
    inst✝⁴ : DecidableEq n
    inst✝³ : DecidableEq p
    inst✝² : CommRing R
    inst✝¹ : Fintype n
    inst✝ : Fintype p
    M : Matrix n n R
    L : List (Matrix.TransvectionStruct n R)
    N : Matrix p p R
    ⊢ Eq (HMul.hMul (Matrix.fromBlocks M 0 0 N) (List.map (Function.comp Matrix.Tr …
  -/
  induction' L with t L IH generalizing M N
    /-
      case nil
      n : Type u_1
      p : Type u_2
      R : Type u₂
      inst✝⁴ : DecidableEq n
      inst✝³ : DecidableEq p
      inst✝² : CommRing R
      inst✝¹ : Fintype n
      inst✝ : Fintype p
      M : Matrix n n R
      N : Matrix p p R
      ⊢ Eq (HMul.hMul (Matrix.fromBlocks M 0 0 N) (List.map (Function.comp Matrix.Tr …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      n : Type u_1
      p : Type u_2
      R : Type u₂
      inst✝⁴ : DecidableEq n
      inst✝³ : DecidableEq p
      inst✝² : CommRing R
      inst✝¹ : Fintype n
      inst✝ : Fintype p
      t : Matrix.TransvectionStruct n R
      L : List (Matrix.TransvectionStruct n R)
      IH : ∀ (M : Matrix n n R) (N : Matrix p p R), Eq (HMul.hMul (Matrix.fromBlocks …
      M : Matrix n n R
      N : Matrix p p R
      ⊢ Eq (HMul.hMul (Matrix.fromBlocks M 0 0 N) (List.map (Function.comp Matrix.Tr …
    -/
  · simp [IH, toMatrix_sumInl, fromBlocks_multiply]
    /-
      🎉 no goals
    -/


/-- Given a `TransvectionStruct` on `n` and an equivalence between `n` and `p`, define the
corresponding `TransvectionStruct` on `p`. -/
def reindexEquiv (e : n ≃ p) (t : TransvectionStruct n R) : TransvectionStruct p R where
  i := e t.i
  j := e t.j
            /-
              n : Type u_1
              p : Type u_2
              R : Type u₂
              𝕜 : Type u_3
              inst✝³ : Field 𝕜
              inst✝² : DecidableEq n
              inst✝¹ : DecidableEq p
              inst✝ : CommRing R
              i j : n
              e : Equiv n p
              t : Matrix.TransvectionStruct n R
              ⊢ Ne (e t.i) (e t.j)
            -/
  hij := by simp [t.hij]
            /-
              🎉 no goals
            -/
  c := t.c


theorem toMatrix_reindexEquiv (e : n ≃ p) (t : TransvectionStruct n R) :
    (t.reindexEquiv e).toMatrix = reindexAlgEquiv R _ e t.toMatrix := by
  /-
    n : Type u_1
    p : Type u_2
    R : Type u₂
    inst✝⁴ : DecidableEq n
    inst✝³ : DecidableEq p
    inst✝² : CommRing R
    inst✝¹ : Fintype n
    inst✝ : Fintype p
    e : Equiv n p
    t : Matrix.TransvectionStruct n R
    ⊢ Eq (Matrix.TransvectionStruct.reindexEquiv e t).toMatrix ((Matrix.reindexAlg …
  -/
  rcases t with ⟨t_i, t_j, _⟩
  /-
    case mk
    n : Type u_1
    p : Type u_2
    R : Type u₂
    inst✝⁴ : DecidableEq n
    inst✝³ : DecidableEq p
    inst✝² : CommRing R
    inst✝¹ : Fintype n
    inst✝ : Fintype p
    e : Equiv n p
    t_i t_j : n
    hij✝ : Ne t_i t_j
    c✝ : R
    ⊢ Eq (Matrix.TransvectionStruct.reindexEquiv e { i := t_i, j := t_j, hij := hi …
  -/
  ext a b
  simp only [reindexEquiv, transvection, mul_boole, Algebra.id.smul_eq_mul, toMatrix_mk,
    submatrix_apply, reindex_apply, DMatrix.add_apply, Pi.smul_apply, reindexAlgEquiv_apply]
  /-
    case mk.a
    n : Type u_1
    p : Type u_2
    R : Type u₂
    inst✝⁴ : DecidableEq n
    inst✝³ : DecidableEq p
    inst✝² : CommRing R
    inst✝¹ : Fintype n
    inst✝ : Fintype p
    e : Equiv n p
    t_i t_j : n
    hij✝ : Ne t_i t_j
    c✝ : R
    a b : p
    ⊢ Eq (HAdd.hAdd 1 (Matrix.stdBasisMatrix (e t_i) (e t_j) c✝) a b) (HAdd.hAdd 1 …
  -/
  by_cases ha : e t_i = a <;> by_cases hb : e t_j = b <;> by_cases hab : a = b <;>
    /-
      case pos
      n : Type u_1
      p : Type u_2
      R : Type u₂
      inst✝⁴ : DecidableEq n
      inst✝³ : DecidableEq p
      inst✝² : CommRing R
      inst✝¹ : Fintype n
      inst✝ : Fintype p
      e : Equiv n p
      t_i t_j : n
      hij✝ : Ne t_i t_j
      c✝ : R
      a b : p
      ha : Eq (e t_i) a
      hb : Eq (e t_j) b
      hab : Eq a b
      ⊢ Eq (HAdd.hAdd 1 (Matrix.stdBasisMatrix (e t_i) (e t_j) c✝) a b) (HAdd.hAdd 1 …
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
    simp [ha, hb, hab, ← e.apply_eq_iff_eq_symm_apply, stdBasisMatrix]
    /-
      🎉 no goals
    -/


theorem toMatrix_reindexEquiv_prod (e : n ≃ p) (L : List (TransvectionStruct n R)) :
    (L.map (toMatrix ∘ reindexEquiv e)).prod = reindexAlgEquiv R _ e (L.map toMatrix).prod := by
  /-
    n : Type u_1
    p : Type u_2
    R : Type u₂
    inst✝⁴ : DecidableEq n
    inst✝³ : DecidableEq p
    inst✝² : CommRing R
    inst✝¹ : Fintype n
    inst✝ : Fintype p
    e : Equiv n p
    L : List (Matrix.TransvectionStruct n R)
    ⊢ Eq (List.map (Function.comp Matrix.TransvectionStruct.toMatrix (Matrix.Trans …
  -/
  induction' L with t L IH
    /-
      case nil
      n : Type u_1
      p : Type u_2
      R : Type u₂
      inst✝⁴ : DecidableEq n
      inst✝³ : DecidableEq p
      inst✝² : CommRing R
      inst✝¹ : Fintype n
      inst✝ : Fintype p
      e : Equiv n p
      ⊢ Eq (List.map (Function.comp Matrix.TransvectionStruct.toMatrix (Matrix.Trans …
    -/
  · simp
    /-
      🎉 no goals
    -/
  · simp only [toMatrix_reindexEquiv, IH, Function.comp_apply, List.prod_cons,
      reindexAlgEquiv_apply, List.map]
    /-
      case cons
      n : Type u_1
      p : Type u_2
      R : Type u₂
      inst✝⁴ : DecidableEq n
      inst✝³ : DecidableEq p
      inst✝² : CommRing R
      inst✝¹ : Fintype n
      inst✝ : Fintype p
      e : Equiv n p
      t : Matrix.TransvectionStruct n R
      L : List (Matrix.TransvectionStruct n R)
      IH : Eq (List.map (Function.comp Matrix.TransvectionStruct.toMatrix (Matrix.Tr …
      ⊢ Eq (HMul.hMul ((Matrix.reindex e e) t.toMatrix) ((Matrix.reindex e e) (List. …
    -/
    exact (reindexAlgEquiv_mul R _ _ _ _).symm
    /-
      🎉 no goals
    -/


/-- A list of transvections such that multiplying on the left with these transvections will replace
the last column with zeroes. -/
def listTransvecCol : List (Matrix (Fin r ⊕ Unit) (Fin r ⊕ Unit) 𝕜) :=
  List.ofFn fun i : Fin r =>
    transvection (inl i) (inr unit) <| -M (inl i) (inr unit) / M (inr unit) (inr unit)


/-- A list of transvections such that multiplying on the right with these transvections will replace
the last row with zeroes. -/
def listTransvecRow : List (Matrix (Fin r ⊕ Unit) (Fin r ⊕ Unit) 𝕜) :=
  List.ofFn fun i : Fin r =>
    transvection (inr unit) (inl i) <| -M (inr unit) (inl i) / M (inr unit) (inr unit)


@[simp]
                                                                      /-
                                                                        𝕜 : Type u_3
                                                                        inst✝ : Field 𝕜
                                                                        r : Nat
                                                                        M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
                                                                        ⊢ Eq (Matrix.Pivot.listTransvecCol M).length r
                                                                      -/
theorem length_listTransvecCol : (listTransvecCol M).length = r := by simp [listTransvecCol]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem listTransvecCol_getElem {i : ℕ} (h : i < (listTransvecCol M).length) :
    (listTransvecCol M)[i] =
      letI i' : Fin r := ⟨i, length_listTransvecCol M ▸ h⟩
      transvection (inl i') (inr unit) <| -M (inl i') (inr unit) / M (inr unit) (inr unit) := by
  /-
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    i : Nat
    h : LT.lt i (Matrix.Pivot.listTransvecCol M).length
    ⊢ Eq (GetElem.getElem (Matrix.Pivot.listTransvecCol M) i h) (Matrix.transvecti …
  -/
  simp [listTransvecCol]
  /-
    🎉 no goals
  -/


@[deprecated listTransvecCol_getElem (since := "2024-08-03")]
theorem listTransvecCol_get (i : Fin (listTransvecCol M).length) :
    (listTransvecCol M).get i =
      letI i' := Fin.cast (length_listTransvecCol M) i
      transvection (inl i') (inr unit) <| -M (inl i') (inr unit) / M (inr unit) (inr unit) :=
  listTransvecCol_getElem _ i.isLt


@[simp]
                                                                      /-
                                                                        𝕜 : Type u_3
                                                                        inst✝ : Field 𝕜
                                                                        r : Nat
                                                                        M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
                                                                        ⊢ Eq (Matrix.Pivot.listTransvecRow M).length r
                                                                      -/
theorem length_listTransvecRow : (listTransvecRow M).length = r := by simp [listTransvecRow]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem listTransvecRow_getElem {i : ℕ} (h : i < (listTransvecRow M).length) :
    (listTransvecRow M)[i] =
      letI i' : Fin r := ⟨i, length_listTransvecRow M ▸ h⟩
      transvection (inr unit) (inl i') <| -M (inr unit) (inl i') / M (inr unit) (inr unit) := by
  /-
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    i : Nat
    h : LT.lt i (Matrix.Pivot.listTransvecRow M).length
    ⊢ Eq (GetElem.getElem (Matrix.Pivot.listTransvecRow M) i h) (Matrix.transvecti …
  -/
  simp [listTransvecRow, Fin.cast]
  /-
    🎉 no goals
  -/


@[deprecated listTransvecRow_getElem (since := "2024-08-03")]
theorem listTransvecRow_get (i : Fin (listTransvecRow M).length) :
    (listTransvecRow M).get i =
      letI i' := Fin.cast (length_listTransvecRow M) i
      transvection (inr unit) (inl i') <| -M (inr unit) (inl i') / M (inr unit) (inr unit) :=
  listTransvecRow_getElem _ i.isLt


/-- Multiplying by some of the matrices in `listTransvecCol M` does not change the last row. -/
theorem listTransvecCol_mul_last_row_drop (i : Fin r ⊕ Unit) {k : ℕ} (hk : k ≤ r) :
    (((listTransvecCol M).drop k).prod * M) (inr unit) i = M (inr unit) i := by
  induction hk using Nat.decreasingInduction with
  | of_succ n hn IH =>
    have hn' : n < (listTransvecCol M).length := by simpa [listTransvecCol] using hn
    rw [List.drop_eq_getElem_cons hn']
    simpa [listTransvecCol, Matrix.mul_assoc]
  | self =>
    simp only [length_listTransvecCol, le_refl, List.drop_eq_nil_of_le, List.prod_nil,
      Matrix.one_mul]


/-- Multiplying by all the matrices in `listTransvecCol M` does not change the last row. -/
theorem listTransvecCol_mul_last_row (i : Fin r ⊕ Unit) :
    ((listTransvecCol M).prod * M) (inr unit) i = M (inr unit) i := by
  /-
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    i : Sum (Fin r) Unit
    ⊢ Eq (HMul.hMul (Matrix.Pivot.listTransvecCol M).prod M (Sum.inr Unit.unit) i) …
  -/
  simpa using listTransvecCol_mul_last_row_drop M i (zero_le _)
  /-
    🎉 no goals
  -/


/-- Multiplying by all the matrices in `listTransvecCol M` kills all the coefficients in the
last column but the last one. -/
theorem listTransvecCol_mul_last_col (hM : M (inr unit) (inr unit) ≠ 0) (i : Fin r) :
    ((listTransvecCol M).prod * M) (inl i) (inr unit) = 0 := by
  suffices H :
    ∀ k : ℕ,
      k ≤ r →
        (((listTransvecCol M).drop k).prod * M) (inl i) (inr unit) =
          if k ≤ i then 0 else M (inl i) (inr unit) by
    simpa only [List.drop, _root_.zero_le, ite_true] using H 0 (zero_le _)
  /-
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
    i : Fin r
    ⊢ ∀ (k : Nat), LE.le k r → Eq (HMul.hMul (List.drop k (Matrix.Pivot.listTransv …
  -/
  intro k hk
  induction hk using Nat.decreasingInduction with
  | of_succ n hn IH =>
    have hn' : n < (listTransvecCol M).length := by simpa [listTransvecCol] using hn
    let n' : Fin r := ⟨n, hn⟩
    rw [List.drop_eq_getElem_cons hn']
    have A :
      (listTransvecCol M)[n] =
        transvection (inl n') (inr unit) (-M (inl n') (inr unit) / M (inr unit) (inr unit)) := by
      simp [n', listTransvecCol]
    simp only [Matrix.mul_assoc, A, List.prod_cons]
    by_cases h : n' = i
    · have hni : n = i := by
        cases i
        simp only [n', Fin.mk_eq_mk] at h
        simp [h]
      simp only [h, transvection_mul_apply_same, IH, ← hni, add_le_iff_nonpos_right,
          listTransvecCol_mul_last_row_drop _ _ hn]
      field_simp [hM]
    · have hni : n ≠ i := by
        rintro rfl
        cases i
        simp [n'] at h
      simp only [ne_eq, inl.injEq, Ne.symm h, not_false_eq_true, transvection_mul_apply_of_ne]
      rw [IH]
      rcases le_or_lt (n + 1) i with (hi | hi)
      · simp only [hi, n.le_succ.trans hi, if_true]
      · rw [if_neg, if_neg]
        · simpa only [hni.symm, not_le, or_false] using Nat.lt_succ_iff_lt_or_eq.1 hi
        · simpa only [not_le] using hi
  | self =>
    simp only [length_listTransvecCol, le_refl, List.drop_eq_nil_of_le, List.prod_nil,
      Matrix.one_mul]
    rw [if_neg]
    simpa only [not_le] using i.2


/-- Multiplying by some of the matrices in `listTransvecRow M` does not change the last column. -/
theorem mul_listTransvecRow_last_col_take (i : Fin r ⊕ Unit) {k : ℕ} (hk : k ≤ r) :
    (M * ((listTransvecRow M).take k).prod) i (inr unit) = M i (inr unit) := by
  /-
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    i : Sum (Fin r) Unit
    k : Nat
    hk : LE.le k r
    ⊢ Eq (HMul.hMul M (List.take k (Matrix.Pivot.listTransvecRow M)).prod i (Sum.i …
  -/
  induction' k with k IH
    /-
      case zero
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      i : Sum (Fin r) Unit
      hk : LE.le 0 r
      ⊢ Eq (HMul.hMul M (List.take 0 (Matrix.Pivot.listTransvecRow M)).prod i (Sum.i …
    -/
  · simp only [Matrix.mul_one, List.take_zero, List.prod_nil, List.take, Matrix.mul_one]
    /-
      🎉 no goals
    -/
    /-
      case succ
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      i : Sum (Fin r) Unit
      k : Nat
      IH : LE.le k r → Eq (HMul.hMul M (List.take k (Matrix.Pivot.listTransvecRow M) …
      hk : LE.le (HAdd.hAdd k 1) r
      ⊢ Eq (HMul.hMul M (List.take (HAdd.hAdd k 1) (Matrix.Pivot.listTransvecRow M)) …
    -/
  · have hkr : k < r := hk
    /-
      case succ
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      i : Sum (Fin r) Unit
      k : Nat
      IH : LE.le k r → Eq (HMul.hMul M (List.take k (Matrix.Pivot.listTransvecRow M) …
      hk : LE.le (HAdd.hAdd k 1) r
      hkr : LT.lt k r
      ⊢ Eq (HMul.hMul M (List.take (HAdd.hAdd k 1) (Matrix.Pivot.listTransvecRow M)) …
    -/
    let k' : Fin r := ⟨k, hkr⟩
    have :
      (listTransvecRow M)[k]? =
        ↑(transvection (inr Unit.unit) (inl k')
            (-M (inr Unit.unit) (inl k') / M (inr Unit.unit) (inr Unit.unit))) := by
      simp only [k', listTransvecRow, List.ofFnNthVal, hkr, dif_pos, List.getElem?_ofFn]
    simp only [List.take_succ, ← Matrix.mul_assoc, this, List.prod_append, Matrix.mul_one,
      List.prod_cons, List.prod_nil, Option.toList_some]
    /-
      case succ
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      i : Sum (Fin r) Unit
      k : Nat
      IH : LE.le k r → Eq (HMul.hMul M (List.take k (Matrix.Pivot.listTransvecRow M) …
      hk : LE.le (HAdd.hAdd k 1) r
      hkr : LT.lt k r
      k' : Fin r := ⟨k, hkr⟩
      this : Eq (GetElem?.getElem? (Matrix.Pivot.listTransvecRow M) k) (Option.some  …
      ⊢ Eq (HMul.hMul (HMul.hMul M (List.take k (Matrix.Pivot.listTransvecRow M)).pr …
    -/
    rw [mul_transvection_apply_of_ne, IH hkr.le]
    /-
      case succ.hb
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      i : Sum (Fin r) Unit
      k : Nat
      IH : LE.le k r → Eq (HMul.hMul M (List.take k (Matrix.Pivot.listTransvecRow M) …
      hk : LE.le (HAdd.hAdd k 1) r
      hkr : LT.lt k r
      k' : Fin r := ⟨k, hkr⟩
      this : Eq (GetElem?.getElem? (Matrix.Pivot.listTransvecRow M) k) (Option.some  …
      ⊢ Ne (Sum.inr Unit.unit) (Sum.inl k')
    -/
    simp only [Ne, not_false_iff, reduceCtorEq]
    /-
      🎉 no goals
    -/


/-- Multiplying by all the matrices in `listTransvecRow M` does not change the last column. -/
theorem mul_listTransvecRow_last_col (i : Fin r ⊕ Unit) :
    (M * (listTransvecRow M).prod) i (inr unit) = M i (inr unit) := by
  /-
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    i : Sum (Fin r) Unit
    ⊢ Eq (HMul.hMul M (Matrix.Pivot.listTransvecRow M).prod i (Sum.inr Unit.unit)) …
  -/
  have A : (listTransvecRow M).length = r := by simp [listTransvecRow]
  /-
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    i : Sum (Fin r) Unit
    A : Eq (Matrix.Pivot.listTransvecRow M).length r
    ⊢ Eq (HMul.hMul M (Matrix.Pivot.listTransvecRow M).prod i (Sum.inr Unit.unit)) …
  -/
  rw [← List.take_length (listTransvecRow M), A]
  /-
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    i : Sum (Fin r) Unit
    A : Eq (Matrix.Pivot.listTransvecRow M).length r
    ⊢ Eq (HMul.hMul M (List.take r (Matrix.Pivot.listTransvecRow M)).prod i (Sum.i …
  -/
  simpa using mul_listTransvecRow_last_col_take M i le_rfl
  /-
    🎉 no goals
  -/


/-- Multiplying by all the matrices in `listTransvecRow M` kills all the coefficients in the
last row but the last one. -/
theorem mul_listTransvecRow_last_row (hM : M (inr unit) (inr unit) ≠ 0) (i : Fin r) :
    (M * (listTransvecRow M).prod) (inr unit) (inl i) = 0 := by
  suffices H :
    ∀ k : ℕ,
      k ≤ r →
        (M * ((listTransvecRow M).take k).prod) (inr unit) (inl i) =
          if k ≤ i then M (inr unit) (inl i) else 0 by
    have A : (listTransvecRow M).length = r := by simp [listTransvecRow]
    rw [← List.take_length (listTransvecRow M), A]
    have : ¬r ≤ i := by simp
    simpa only [this, ite_eq_right_iff] using H r le_rfl
  /-
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
    i : Fin r
    ⊢ ∀ (k : Nat), LE.le k r → Eq (HMul.hMul M (List.take k (Matrix.Pivot.listTran …
  -/
  intro k hk
  /-
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
    i : Fin r
    k : Nat
    hk : LE.le k r
    ⊢ Eq (HMul.hMul M (List.take k (Matrix.Pivot.listTransvecRow M)).prod (Sum.inr …
  -/
  induction' k with n IH
    /-
      case zero
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      i : Fin r
      hk : LE.le 0 r
      ⊢ Eq (HMul.hMul M (List.take 0 (Matrix.Pivot.listTransvecRow M)).prod (Sum.inr …
    -/
  · simp only [if_true, Matrix.mul_one, List.take_zero, zero_le', List.prod_nil]
    /-
      🎉 no goals
    -/
    /-
      case succ
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      i : Fin r
      n : Nat
      IH : LE.le n r → Eq (HMul.hMul M (List.take n (Matrix.Pivot.listTransvecRow M) …
      hk : LE.le (HAdd.hAdd n 1) r
      ⊢ Eq (HMul.hMul M (List.take (HAdd.hAdd n 1) (Matrix.Pivot.listTransvecRow M)) …
    -/
  · have hnr : n < r := hk
    /-
      case succ
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      i : Fin r
      n : Nat
      IH : LE.le n r → Eq (HMul.hMul M (List.take n (Matrix.Pivot.listTransvecRow M) …
      hk : LE.le (HAdd.hAdd n 1) r
      hnr : LT.lt n r
      ⊢ Eq (HMul.hMul M (List.take (HAdd.hAdd n 1) (Matrix.Pivot.listTransvecRow M)) …
    -/
    let n' : Fin r := ⟨n, hnr⟩
    have A :
      (listTransvecRow M)[n]? =
        ↑(transvection (inr unit) (inl n')
        (-M (inr unit) (inl n') / M (inr unit) (inr unit))) := by
      simp only [n', listTransvecRow, List.ofFnNthVal, hnr, dif_pos, List.getElem?_ofFn]
    simp only [List.take_succ, A, ← Matrix.mul_assoc, List.prod_append, Matrix.mul_one,
      List.prod_cons, List.prod_nil, Option.toList_some]
    /-
      case succ
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      i : Fin r
      n : Nat
      IH : LE.le n r → Eq (HMul.hMul M (List.take n (Matrix.Pivot.listTransvecRow M) …
      hk : LE.le (HAdd.hAdd n 1) r
      hnr : LT.lt n r
      n' : Fin r := ⟨n, hnr⟩
      A : Eq (GetElem?.getElem? (Matrix.Pivot.listTransvecRow M) n) (Option.some (Ma …
      ⊢ Eq (HMul.hMul (HMul.hMul M (List.take n (Matrix.Pivot.listTransvecRow M)).pr …
    -/
    by_cases h : n' = i
    · have hni : n = i := by
        cases i
        simp only [n', Fin.mk_eq_mk] at h
        simp only [h]
      /-
        case pos
        𝕜 : Type u_3
        inst✝ : Field 𝕜
        r : Nat
        M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
        hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
        i : Fin r
        n : Nat
        IH : LE.le n r → Eq (HMul.hMul M (List.take n (Matrix.Pivot.listTransvecRow M) …
        hk : LE.le (HAdd.hAdd n 1) r
        hnr : LT.lt n r
        n' : Fin r := ⟨n, hnr⟩
        A : Eq (GetElem?.getElem? (Matrix.Pivot.listTransvecRow M) n) (Option.some (Ma …
        h : Eq n' i
        hni : Eq n ↑i
        ⊢ Eq (HMul.hMul (HMul.hMul M (List.take n (Matrix.Pivot.listTransvecRow M)).pr …
      -/
      have : ¬n.succ ≤ i := by simp only [← hni, n.lt_succ_self, not_le]
      simp only [h, mul_transvection_apply_same, List.take, if_false,
        mul_listTransvecRow_last_col_take _ _ hnr.le, hni.le, this, if_true, IH hnr.le]
      /-
        case pos
        𝕜 : Type u_3
        inst✝ : Field 𝕜
        r : Nat
        M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
        hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
        i : Fin r
        n : Nat
        IH : LE.le n r → Eq (HMul.hMul M (List.take n (Matrix.Pivot.listTransvecRow M) …
        hk : LE.le (HAdd.hAdd n 1) r
        hnr : LT.lt n r
        n' : Fin r := ⟨n, hnr⟩
        A : Eq (GetElem?.getElem? (Matrix.Pivot.listTransvecRow M) n) (Option.some (Ma …
        h : Eq n' i
        hni : Eq n ↑i
        this : Not (LE.le n.succ ↑i)
        ⊢ Eq (HAdd.hAdd (M (Sum.inr Unit.unit) (Sum.inl i)) (HMul.hMul (HDiv.hDiv (Neg …
      -/
      field_simp [hM]
      /-
        🎉 no goals
      -/
    · have hni : n ≠ i := by
        rintro rfl
        cases i
        tauto
      simp only [IH hnr.le, Ne, mul_transvection_apply_of_ne, Ne.symm h, inl.injEq,
        not_false_eq_true]
      /-
        case neg
        𝕜 : Type u_3
        inst✝ : Field 𝕜
        r : Nat
        M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
        hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
        i : Fin r
        n : Nat
        IH : LE.le n r → Eq (HMul.hMul M (List.take n (Matrix.Pivot.listTransvecRow M) …
        hk : LE.le (HAdd.hAdd n 1) r
        hnr : LT.lt n r
        n' : Fin r := ⟨n, hnr⟩
        A : Eq (GetElem?.getElem? (Matrix.Pivot.listTransvecRow M) n) (Option.some (Ma …
        h : Not (Eq n' i)
        hni : Ne n ↑i
        ⊢ Eq (ite (LE.le n ↑i) (M (Sum.inr Unit.unit) (Sum.inl i)) 0) (ite (LE.le (HAd …
      -/
      rcases le_or_lt (n + 1) i with (hi | hi)
        /-
          case neg.inl
          𝕜 : Type u_3
          inst✝ : Field 𝕜
          r : Nat
          M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
          hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
          i : Fin r
          n : Nat
          IH : LE.le n r → Eq (HMul.hMul M (List.take n (Matrix.Pivot.listTransvecRow M) …
          hk : LE.le (HAdd.hAdd n 1) r
          hnr : LT.lt n r
          n' : Fin r := ⟨n, hnr⟩
          A : Eq (GetElem?.getElem? (Matrix.Pivot.listTransvecRow M) n) (Option.some (Ma …
          h : Not (Eq n' i)
          hni : Ne n ↑i
          hi : LE.le (HAdd.hAdd n 1) ↑i
          ⊢ Eq (ite (LE.le n ↑i) (M (Sum.inr Unit.unit) (Sum.inl i)) 0) (ite (LE.le (HAd …
        -/
      · simp [hi, n.le_succ.trans hi, if_true]
        /-
          🎉 no goals
        -/
        /-
          case neg.inr
          𝕜 : Type u_3
          inst✝ : Field 𝕜
          r : Nat
          M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
          hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
          i : Fin r
          n : Nat
          IH : LE.le n r → Eq (HMul.hMul M (List.take n (Matrix.Pivot.listTransvecRow M) …
          hk : LE.le (HAdd.hAdd n 1) r
          hnr : LT.lt n r
          n' : Fin r := ⟨n, hnr⟩
          A : Eq (GetElem?.getElem? (Matrix.Pivot.listTransvecRow M) n) (Option.some (Ma …
          h : Not (Eq n' i)
          hni : Ne n ↑i
          hi : LT.lt (↑i) (HAdd.hAdd n 1)
          ⊢ Eq (ite (LE.le n ↑i) (M (Sum.inr Unit.unit) (Sum.inl i)) 0) (ite (LE.le (HAd …
        -/
      · rw [if_neg, if_neg]
          /-
            case neg.inr.hnc
            𝕜 : Type u_3
            inst✝ : Field 𝕜
            r : Nat
            M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
            hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
            i : Fin r
            n : Nat
            IH : LE.le n r → Eq (HMul.hMul M (List.take n (Matrix.Pivot.listTransvecRow M) …
            hk : LE.le (HAdd.hAdd n 1) r
            hnr : LT.lt n r
            n' : Fin r := ⟨n, hnr⟩
            A : Eq (GetElem?.getElem? (Matrix.Pivot.listTransvecRow M) n) (Option.some (Ma …
            h : Not (Eq n' i)
            hni : Ne n ↑i
            hi : LT.lt (↑i) (HAdd.hAdd n 1)
            ⊢ Not (LE.le (HAdd.hAdd n 1) ↑i)
          -/
        · simpa only [not_le] using hi
          /-
            🎉 no goals
          -/
          /-
            case neg.inr.hnc
            𝕜 : Type u_3
            inst✝ : Field 𝕜
            r : Nat
            M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
            hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
            i : Fin r
            n : Nat
            IH : LE.le n r → Eq (HMul.hMul M (List.take n (Matrix.Pivot.listTransvecRow M) …
            hk : LE.le (HAdd.hAdd n 1) r
            hnr : LT.lt n r
            n' : Fin r := ⟨n, hnr⟩
            A : Eq (GetElem?.getElem? (Matrix.Pivot.listTransvecRow M) n) (Option.some (Ma …
            h : Not (Eq n' i)
            hni : Ne n ↑i
            hi : LT.lt (↑i) (HAdd.hAdd n 1)
            ⊢ Not (LE.le n ↑i)
          -/
        · simpa only [hni.symm, not_le, or_false] using Nat.lt_succ_iff_lt_or_eq.1 hi
          /-
            🎉 no goals
          -/


/-- Multiplying by all the matrices either in `listTransvecCol M` and `listTransvecRow M` kills
all the coefficients in the last row but the last one. -/
theorem listTransvecCol_mul_mul_listTransvecRow_last_col (hM : M (inr unit) (inr unit) ≠ 0)
    (i : Fin r) :
    ((listTransvecCol M).prod * M * (listTransvecRow M).prod) (inr unit) (inl i) = 0 := by
  have : listTransvecRow M = listTransvecRow ((listTransvecCol M).prod * M) := by
    simp [listTransvecRow, listTransvecCol_mul_last_row]
  /-
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
    i : Fin r
    this : Eq (Matrix.Pivot.listTransvecRow M) (Matrix.Pivot.listTransvecRow (HMul …
    ⊢ Eq (HMul.hMul (HMul.hMul (Matrix.Pivot.listTransvecCol M).prod M) (Matrix.Pi …
  -/
  rw [this]
  /-
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
    i : Fin r
    this : Eq (Matrix.Pivot.listTransvecRow M) (Matrix.Pivot.listTransvecRow (HMul …
    ⊢ Eq (HMul.hMul (HMul.hMul (Matrix.Pivot.listTransvecCol M).prod M) (Matrix.Pi …
  -/
  apply mul_listTransvecRow_last_row
  /-
    case hM
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
    i : Fin r
    this : Eq (Matrix.Pivot.listTransvecRow M) (Matrix.Pivot.listTransvecRow (HMul …
    ⊢ Ne (HMul.hMul (Matrix.Pivot.listTransvecCol M).prod M (Sum.inr Unit.unit) (S …
  -/
  simpa [listTransvecCol_mul_last_row] using hM
  /-
    🎉 no goals
  -/


/-- Multiplying by all the matrices either in `listTransvecCol M` and `listTransvecRow M` kills
all the coefficients in the last column but the last one. -/
theorem listTransvecCol_mul_mul_listTransvecRow_last_row (hM : M (inr unit) (inr unit) ≠ 0)
    (i : Fin r) :
    ((listTransvecCol M).prod * M * (listTransvecRow M).prod) (inl i) (inr unit) = 0 := by
  have : listTransvecCol M = listTransvecCol (M * (listTransvecRow M).prod) := by
    simp [listTransvecCol, mul_listTransvecRow_last_col]
  /-
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
    i : Fin r
    this : Eq (Matrix.Pivot.listTransvecCol M) (Matrix.Pivot.listTransvecCol (HMul …
    ⊢ Eq (HMul.hMul (HMul.hMul (Matrix.Pivot.listTransvecCol M).prod M) (Matrix.Pi …
  -/
  rw [this, Matrix.mul_assoc]
  /-
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
    i : Fin r
    this : Eq (Matrix.Pivot.listTransvecCol M) (Matrix.Pivot.listTransvecCol (HMul …
    ⊢ Eq (HMul.hMul (Matrix.Pivot.listTransvecCol (HMul.hMul M (Matrix.Pivot.listT …
  -/
  apply listTransvecCol_mul_last_col
  /-
    case hM
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
    i : Fin r
    this : Eq (Matrix.Pivot.listTransvecCol M) (Matrix.Pivot.listTransvecCol (HMul …
    ⊢ Ne (HMul.hMul M (Matrix.Pivot.listTransvecRow M).prod (Sum.inr Unit.unit) (S …
  -/
  simpa [mul_listTransvecRow_last_col] using hM
  /-
    🎉 no goals
  -/


/-- Multiplying by all the matrices either in `listTransvecCol M` and `listTransvecRow M` turns
the matrix in block-diagonal form. -/
theorem isTwoBlockDiagonal_listTransvecCol_mul_mul_listTransvecRow
    (hM : M (inr unit) (inr unit) ≠ 0) :
    IsTwoBlockDiagonal ((listTransvecCol M).prod * M * (listTransvecRow M).prod) := by
  /-
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
    ⊢ (HMul.hMul (HMul.hMul (Matrix.Pivot.listTransvecCol M).prod M) (Matrix.Pivot …
  -/
  constructor
    /-
      case left
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      ⊢ Eq (HMul.hMul (HMul.hMul (Matrix.Pivot.listTransvecCol M).prod M) (Matrix.Pi …
    -/
  · ext i j
    /-
      case left.a
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      i : Fin r
      j : Unit
      ⊢ Eq ((HMul.hMul (HMul.hMul (Matrix.Pivot.listTransvecCol M).prod M) (Matrix.P …
    -/
    have : j = unit := by simp only [eq_iff_true_of_subsingleton]
    /-
      case left.a
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      i : Fin r
      j : Unit
      this : Eq j Unit.unit
      ⊢ Eq ((HMul.hMul (HMul.hMul (Matrix.Pivot.listTransvecCol M).prod M) (Matrix.P …
    -/
    simp [toBlocks₁₂, this, listTransvecCol_mul_mul_listTransvecRow_last_row M hM]
    /-
      🎉 no goals
    -/
    /-
      case right
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      ⊢ Eq (HMul.hMul (HMul.hMul (Matrix.Pivot.listTransvecCol M).prod M) (Matrix.Pi …
    -/
  · ext i j
    /-
      case right.a
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      i : Unit
      j : Fin r
      ⊢ Eq ((HMul.hMul (HMul.hMul (Matrix.Pivot.listTransvecCol M).prod M) (Matrix.P …
    -/
    have : i = unit := by simp only [eq_iff_true_of_subsingleton]
    /-
      case right.a
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      i : Unit
      j : Fin r
      this : Eq i Unit.unit
      ⊢ Eq ((HMul.hMul (HMul.hMul (Matrix.Pivot.listTransvecCol M).prod M) (Matrix.P …
    -/
    simp [toBlocks₂₁, this, listTransvecCol_mul_mul_listTransvecRow_last_col M hM]
    /-
      🎉 no goals
    -/


/-- There exist two lists of `TransvectionStruct` such that multiplying by them on the left and
on the right makes a matrix block-diagonal, when the last coefficient is nonzero. -/
theorem exists_isTwoBlockDiagonal_of_ne_zero (hM : M (inr unit) (inr unit) ≠ 0) :
    ∃ L L' : List (TransvectionStruct (Fin r ⊕ Unit) 𝕜),
      IsTwoBlockDiagonal ((L.map toMatrix).prod * M * (L'.map toMatrix).prod) := by
  let L : List (TransvectionStruct (Fin r ⊕ Unit) 𝕜) :=
    List.ofFn fun i : Fin r =>
      ⟨inl i, inr unit, by simp, -M (inl i) (inr unit) / M (inr unit) (inr unit)⟩
  let L' : List (TransvectionStruct (Fin r ⊕ Unit) 𝕜) :=
    List.ofFn fun i : Fin r =>
      ⟨inr unit, inl i, by simp, -M (inr unit) (inl i) / M (inr unit) (inr unit)⟩
  /-
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
    L : List (Matrix.TransvectionStruct (Sum (Fin r) Unit) 𝕜) := List.ofFn fun i = …
    L' : List (Matrix.TransvectionStruct (Sum (Fin r) Unit) 𝕜) := List.ofFn fun i  …
    ⊢ Exists fun L => Exists fun L' => (HMul.hMul (HMul.hMul (List.map Matrix.Tran …
  -/
  refine ⟨L, L', ?_⟩
  /-
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
    L : List (Matrix.TransvectionStruct (Sum (Fin r) Unit) 𝕜) := List.ofFn fun i = …
    L' : List (Matrix.TransvectionStruct (Sum (Fin r) Unit) 𝕜) := List.ofFn fun i  …
    ⊢ (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L).prod M …
  -/
  have A : L.map toMatrix = listTransvecCol M := by simp [L, listTransvecCol, Function.comp_def]
  /-
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
    L : List (Matrix.TransvectionStruct (Sum (Fin r) Unit) 𝕜) := List.ofFn fun i = …
    L' : List (Matrix.TransvectionStruct (Sum (Fin r) Unit) 𝕜) := List.ofFn fun i  …
    A : Eq (List.map Matrix.TransvectionStruct.toMatrix L) (Matrix.Pivot.listTrans …
    ⊢ (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L).prod M …
  -/
  have B : L'.map toMatrix = listTransvecRow M := by simp [L', listTransvecRow, Function.comp_def]
  /-
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
    L : List (Matrix.TransvectionStruct (Sum (Fin r) Unit) 𝕜) := List.ofFn fun i = …
    L' : List (Matrix.TransvectionStruct (Sum (Fin r) Unit) 𝕜) := List.ofFn fun i  …
    A : Eq (List.map Matrix.TransvectionStruct.toMatrix L) (Matrix.Pivot.listTrans …
    B : Eq (List.map Matrix.TransvectionStruct.toMatrix L') (Matrix.Pivot.listTran …
    ⊢ (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L).prod M …
  -/
  rw [A, B]
  /-
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
    L : List (Matrix.TransvectionStruct (Sum (Fin r) Unit) 𝕜) := List.ofFn fun i = …
    L' : List (Matrix.TransvectionStruct (Sum (Fin r) Unit) 𝕜) := List.ofFn fun i  …
    A : Eq (List.map Matrix.TransvectionStruct.toMatrix L) (Matrix.Pivot.listTrans …
    B : Eq (List.map Matrix.TransvectionStruct.toMatrix L') (Matrix.Pivot.listTran …
    ⊢ (HMul.hMul (HMul.hMul (Matrix.Pivot.listTransvecCol M).prod M) (Matrix.Pivot …
  -/
  exact isTwoBlockDiagonal_listTransvecCol_mul_mul_listTransvecRow M hM
  /-
    🎉 no goals
  -/


/-- There exist two lists of `TransvectionStruct` such that multiplying by them on the left and
on the right makes a matrix block-diagonal. -/
theorem exists_isTwoBlockDiagonal_list_transvec_mul_mul_list_transvec
    (M : Matrix (Fin r ⊕ Unit) (Fin r ⊕ Unit) 𝕜) :
    ∃ L L' : List (TransvectionStruct (Fin r ⊕ Unit) 𝕜),
      IsTwoBlockDiagonal ((L.map toMatrix).prod * M * (L'.map toMatrix).prod) := by
  /-
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    ⊢ Exists fun L => Exists fun L' => (HMul.hMul (HMul.hMul (List.map Matrix.Tran …
  -/
  by_cases H : IsTwoBlockDiagonal M
    /-
      case pos
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      H : M.IsTwoBlockDiagonal
      ⊢ Exists fun L => Exists fun L' => (HMul.hMul (HMul.hMul (List.map Matrix.Tran …
    -/
  · refine ⟨List.nil, List.nil, by simpa using H⟩
    /-
      🎉 no goals
    -/
  -- we have already proved this when the last coefficient is nonzero
  /-
    case neg
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    H : Not M.IsTwoBlockDiagonal
    ⊢ Exists fun L => Exists fun L' => (HMul.hMul (HMul.hMul (List.map Matrix.Tran …
  -/
  by_cases hM : M (inr unit) (inr unit) ≠ 0
    /-
      case pos
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      H : Not M.IsTwoBlockDiagonal
      hM : Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      ⊢ Exists fun L => Exists fun L' => (HMul.hMul (HMul.hMul (List.map Matrix.Tran …
    -/
  · exact exists_isTwoBlockDiagonal_of_ne_zero M hM
    /-
      🎉 no goals
    -/
  -- when the last coefficient is zero but there is a nonzero coefficient on the last row or the
  -- last column, we will first put this nonzero coefficient in last position, and then argue as
  -- above.
  /-
    case neg
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    H : Not M.IsTwoBlockDiagonal
    hM : Not (Ne (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0)
    ⊢ Exists fun L => Exists fun L' => (HMul.hMul (HMul.hMul (List.map Matrix.Tran …
  -/
  push_neg at hM
  /-
    case neg
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    H : Not M.IsTwoBlockDiagonal
    hM : Eq (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
    ⊢ Exists fun L => Exists fun L' => (HMul.hMul (HMul.hMul (List.map Matrix.Tran …
  -/
  simp only [not_and_or, IsTwoBlockDiagonal, toBlocks₁₂, toBlocks₂₁, ← Matrix.ext_iff] at H
  have : ∃ i : Fin r, M (inl i) (inr unit) ≠ 0 ∨ M (inr unit) (inl i) ≠ 0 := by
    cases' H with H H
    · contrapose! H
      rintro i ⟨⟩
      exact (H i).1
    · contrapose! H
      rintro ⟨⟩ j
      exact (H j).2
  /-
    case neg
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    hM : Eq (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
    H : Or (Not (∀ (i : Fin r) (j : Unit), Eq (Matrix.of (fun i j => M (Sum.inl i) …
    this : Exists fun i => Or (Ne (M (Sum.inl i) (Sum.inr Unit.unit)) 0) (Ne (M (S …
    ⊢ Exists fun L => Exists fun L' => (HMul.hMul (HMul.hMul (List.map Matrix.Tran …
  -/
  rcases this with ⟨i, h | h⟩
    /-
      case neg.intro.inl
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      hM : Eq (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      H : Or (Not (∀ (i : Fin r) (j : Unit), Eq (Matrix.of (fun i j => M (Sum.inl i) …
      i : Fin r
      h : Ne (M (Sum.inl i) (Sum.inr Unit.unit)) 0
      ⊢ Exists fun L => Exists fun L' => (HMul.hMul (HMul.hMul (List.map Matrix.Tran …
    -/
  · let M' := transvection (inr Unit.unit) (inl i) 1 * M
    /-
      case neg.intro.inl
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      hM : Eq (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      H : Or (Not (∀ (i : Fin r) (j : Unit), Eq (Matrix.of (fun i j => M (Sum.inl i) …
      i : Fin r
      h : Ne (M (Sum.inl i) (Sum.inr Unit.unit)) 0
      M' : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜 := HMul.hMul (Matrix.trans …
      ⊢ Exists fun L => Exists fun L' => (HMul.hMul (HMul.hMul (List.map Matrix.Tran …
    -/
    have hM' : M' (inr unit) (inr unit) ≠ 0 := by simpa [M', hM]
    /-
      case neg.intro.inl
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      hM : Eq (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      H : Or (Not (∀ (i : Fin r) (j : Unit), Eq (Matrix.of (fun i j => M (Sum.inl i) …
      i : Fin r
      h : Ne (M (Sum.inl i) (Sum.inr Unit.unit)) 0
      M' : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜 := HMul.hMul (Matrix.trans …
      hM' : Ne (M' (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      ⊢ Exists fun L => Exists fun L' => (HMul.hMul (HMul.hMul (List.map Matrix.Tran …
    -/
    rcases exists_isTwoBlockDiagonal_of_ne_zero M' hM' with ⟨L, L', hLL'⟩
    /-
      case neg.intro.inl.intro.intro
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      hM : Eq (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      H : Or (Not (∀ (i : Fin r) (j : Unit), Eq (Matrix.of (fun i j => M (Sum.inl i) …
      i : Fin r
      h : Ne (M (Sum.inl i) (Sum.inr Unit.unit)) 0
      M' : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜 := HMul.hMul (Matrix.trans …
      hM' : Ne (M' (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      L L' : List (Matrix.TransvectionStruct (Sum (Fin r) Unit) 𝕜)
      hLL' : (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L).p …
      ⊢ Exists fun L => Exists fun L' => (HMul.hMul (HMul.hMul (List.map Matrix.Tran …
    -/
    rw [Matrix.mul_assoc] at hLL'
    /-
      case neg.intro.inl.intro.intro
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      hM : Eq (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      H : Or (Not (∀ (i : Fin r) (j : Unit), Eq (Matrix.of (fun i j => M (Sum.inl i) …
      i : Fin r
      h : Ne (M (Sum.inl i) (Sum.inr Unit.unit)) 0
      M' : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜 := HMul.hMul (Matrix.trans …
      hM' : Ne (M' (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      L L' : List (Matrix.TransvectionStruct (Sum (Fin r) Unit) 𝕜)
      hLL' : (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L).prod (HMul.h …
      ⊢ Exists fun L => Exists fun L' => (HMul.hMul (HMul.hMul (List.map Matrix.Tran …
    -/
    refine ⟨L ++ [⟨inr unit, inl i, by simp, 1⟩], L', ?_⟩
    simp only [List.map_append, List.prod_append, Matrix.mul_one, toMatrix_mk, List.prod_cons,
      List.prod_nil, List.map, Matrix.mul_assoc (L.map toMatrix).prod]
    /-
      case neg.intro.inl.intro.intro
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      hM : Eq (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      H : Or (Not (∀ (i : Fin r) (j : Unit), Eq (Matrix.of (fun i j => M (Sum.inl i) …
      i : Fin r
      h : Ne (M (Sum.inl i) (Sum.inr Unit.unit)) 0
      M' : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜 := HMul.hMul (Matrix.trans …
      hM' : Ne (M' (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      L L' : List (Matrix.TransvectionStruct (Sum (Fin r) Unit) 𝕜)
      hLL' : (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L).prod (HMul.h …
      ⊢ (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L).prod (HMul.hMul ( …
    -/
    exact hLL'
    /-
      🎉 no goals
    -/
    /-
      case neg.intro.inr
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      hM : Eq (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      H : Or (Not (∀ (i : Fin r) (j : Unit), Eq (Matrix.of (fun i j => M (Sum.inl i) …
      i : Fin r
      h : Ne (M (Sum.inr Unit.unit) (Sum.inl i)) 0
      ⊢ Exists fun L => Exists fun L' => (HMul.hMul (HMul.hMul (List.map Matrix.Tran …
    -/
  · let M' := M * transvection (inl i) (inr unit) 1
    /-
      case neg.intro.inr
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      hM : Eq (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      H : Or (Not (∀ (i : Fin r) (j : Unit), Eq (Matrix.of (fun i j => M (Sum.inl i) …
      i : Fin r
      h : Ne (M (Sum.inr Unit.unit) (Sum.inl i)) 0
      M' : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜 := HMul.hMul M (Matrix.tra …
      ⊢ Exists fun L => Exists fun L' => (HMul.hMul (HMul.hMul (List.map Matrix.Tran …
    -/
    have hM' : M' (inr unit) (inr unit) ≠ 0 := by simpa [M', hM]
    /-
      case neg.intro.inr
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      hM : Eq (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      H : Or (Not (∀ (i : Fin r) (j : Unit), Eq (Matrix.of (fun i j => M (Sum.inl i) …
      i : Fin r
      h : Ne (M (Sum.inr Unit.unit) (Sum.inl i)) 0
      M' : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜 := HMul.hMul M (Matrix.tra …
      hM' : Ne (M' (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      ⊢ Exists fun L => Exists fun L' => (HMul.hMul (HMul.hMul (List.map Matrix.Tran …
    -/
    rcases exists_isTwoBlockDiagonal_of_ne_zero M' hM' with ⟨L, L', hLL'⟩
    /-
      case neg.intro.inr.intro.intro
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      hM : Eq (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      H : Or (Not (∀ (i : Fin r) (j : Unit), Eq (Matrix.of (fun i j => M (Sum.inl i) …
      i : Fin r
      h : Ne (M (Sum.inr Unit.unit) (Sum.inl i)) 0
      M' : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜 := HMul.hMul M (Matrix.tra …
      hM' : Ne (M' (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      L L' : List (Matrix.TransvectionStruct (Sum (Fin r) Unit) 𝕜)
      hLL' : (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L).p …
      ⊢ Exists fun L => Exists fun L' => (HMul.hMul (HMul.hMul (List.map Matrix.Tran …
    -/
    refine ⟨L, ⟨inl i, inr unit, by simp, 1⟩::L', ?_⟩
    /-
      case neg.intro.inr.intro.intro
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      hM : Eq (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      H : Or (Not (∀ (i : Fin r) (j : Unit), Eq (Matrix.of (fun i j => M (Sum.inl i) …
      i : Fin r
      h : Ne (M (Sum.inr Unit.unit) (Sum.inl i)) 0
      M' : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜 := HMul.hMul M (Matrix.tra …
      hM' : Ne (M' (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      L L' : List (Matrix.TransvectionStruct (Sum (Fin r) Unit) 𝕜)
      hLL' : (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L).p …
      ⊢ (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L).prod M …
    -/
    simp only [← Matrix.mul_assoc, toMatrix_mk, List.prod_cons, List.map]
    /-
      case neg.intro.inr.intro.intro
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      hM : Eq (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      H : Or (Not (∀ (i : Fin r) (j : Unit), Eq (Matrix.of (fun i j => M (Sum.inl i) …
      i : Fin r
      h : Ne (M (Sum.inr Unit.unit) (Sum.inl i)) 0
      M' : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜 := HMul.hMul M (Matrix.tra …
      hM' : Ne (M' (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      L L' : List (Matrix.TransvectionStruct (Sum (Fin r) Unit) 𝕜)
      hLL' : (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L).p …
      ⊢ (HMul.hMul (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatri …
    -/
    rw [Matrix.mul_assoc (L.map toMatrix).prod]
    /-
      case neg.intro.inr.intro.intro
      𝕜 : Type u_3
      inst✝ : Field 𝕜
      r : Nat
      M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
      hM : Eq (M (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      H : Or (Not (∀ (i : Fin r) (j : Unit), Eq (Matrix.of (fun i j => M (Sum.inl i) …
      i : Fin r
      h : Ne (M (Sum.inr Unit.unit) (Sum.inl i)) 0
      M' : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜 := HMul.hMul M (Matrix.tra …
      hM' : Ne (M' (Sum.inr Unit.unit) (Sum.inr Unit.unit)) 0
      L L' : List (Matrix.TransvectionStruct (Sum (Fin r) Unit) 𝕜)
      hLL' : (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L).p …
      ⊢ (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L).prod ( …
    -/
    exact hLL'
    /-
      🎉 no goals
    -/


/-- Inductive step for the reduction: if one knows that any size `r` matrix can be reduced to
diagonal form by elementary operations, then one deduces it for matrices over `Fin r ⊕ Unit`. -/
theorem exists_list_transvec_mul_mul_list_transvec_eq_diagonal_induction
    (IH :
      ∀ M : Matrix (Fin r) (Fin r) 𝕜,
        ∃ (L₀ L₀' : List (TransvectionStruct (Fin r) 𝕜)) (D₀ : Fin r → 𝕜),
          (L₀.map toMatrix).prod * M * (L₀'.map toMatrix).prod = diagonal D₀)
    (M : Matrix (Fin r ⊕ Unit) (Fin r ⊕ Unit) 𝕜) :
    ∃ (L L' : List (TransvectionStruct (Fin r ⊕ Unit) 𝕜)) (D : Fin r ⊕ Unit → 𝕜),
      (L.map toMatrix).prod * M * (L'.map toMatrix).prod = diagonal D := by
  /-
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    IH : ∀ (M : Matrix (Fin r) (Fin r) 𝕜), Exists fun L₀ => Exists fun L₀' => Exis …
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    ⊢ Exists fun L => Exists fun L' => Exists fun D => Eq (HMul.hMul (HMul.hMul (L …
  -/
  rcases exists_isTwoBlockDiagonal_list_transvec_mul_mul_list_transvec M with ⟨L₁, L₁', hM⟩
  /-
    case intro.intro
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    IH : ∀ (M : Matrix (Fin r) (Fin r) 𝕜), Exists fun L₀ => Exists fun L₀' => Exis …
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    L₁ L₁' : List (Matrix.TransvectionStruct (Sum (Fin r) Unit) 𝕜)
    hM : (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L₁).pr …
    ⊢ Exists fun L => Exists fun L' => Exists fun D => Eq (HMul.hMul (HMul.hMul (L …
  -/
  let M' := (L₁.map toMatrix).prod * M * (L₁'.map toMatrix).prod
  /-
    case intro.intro
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    IH : ∀ (M : Matrix (Fin r) (Fin r) 𝕜), Exists fun L₀ => Exists fun L₀' => Exis …
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    L₁ L₁' : List (Matrix.TransvectionStruct (Sum (Fin r) Unit) 𝕜)
    hM : (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L₁).pr …
    M' : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜 := HMul.hMul (HMul.hMul (L …
    ⊢ Exists fun L => Exists fun L' => Exists fun D => Eq (HMul.hMul (HMul.hMul (L …
  -/
  let M'' := toBlocks₁₁ M'
  /-
    case intro.intro
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    IH : ∀ (M : Matrix (Fin r) (Fin r) 𝕜), Exists fun L₀ => Exists fun L₀' => Exis …
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    L₁ L₁' : List (Matrix.TransvectionStruct (Sum (Fin r) Unit) 𝕜)
    hM : (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L₁).pr …
    M' : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜 := HMul.hMul (HMul.hMul (L …
    M'' : Matrix (Fin r) (Fin r) 𝕜 := M'.toBlocks₁₁
    ⊢ Exists fun L => Exists fun L' => Exists fun D => Eq (HMul.hMul (HMul.hMul (L …
  -/
  rcases IH M'' with ⟨L₀, L₀', D₀, h₀⟩
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    IH : ∀ (M : Matrix (Fin r) (Fin r) 𝕜), Exists fun L₀ => Exists fun L₀' => Exis …
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    L₁ L₁' : List (Matrix.TransvectionStruct (Sum (Fin r) Unit) 𝕜)
    hM : (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L₁).pr …
    M' : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜 := HMul.hMul (HMul.hMul (L …
    M'' : Matrix (Fin r) (Fin r) 𝕜 := M'.toBlocks₁₁
    L₀ L₀' : List (Matrix.TransvectionStruct (Fin r) 𝕜)
    D₀ : Fin r → 𝕜
    h₀ : Eq (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L₀) …
    ⊢ Exists fun L => Exists fun L' => Exists fun D => Eq (HMul.hMul (HMul.hMul (L …
  -/
  set c := M' (inr unit) (inr unit)
  refine
    ⟨L₀.map (sumInl Unit) ++ L₁, L₁' ++ L₀'.map (sumInl Unit),
      Sum.elim D₀ fun _ => M' (inr unit) (inr unit), ?_⟩
  suffices (L₀.map (toMatrix ∘ sumInl Unit)).prod * M' * (L₀'.map (toMatrix ∘ sumInl Unit)).prod =
      diagonal (Sum.elim D₀ fun _ => c) by
    simpa [M', c, Matrix.mul_assoc]
  have : M' = fromBlocks M'' 0 0 (diagonal fun _ => c) := by
    -- Porting note: simplified proof, because `congr` didn't work anymore
    rw [← fromBlocks_toBlocks M', hM.1, hM.2]
    rfl
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    IH : ∀ (M : Matrix (Fin r) (Fin r) 𝕜), Exists fun L₀ => Exists fun L₀' => Exis …
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    L₁ L₁' : List (Matrix.TransvectionStruct (Sum (Fin r) Unit) 𝕜)
    hM : (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L₁).pr …
    M' : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜 := HMul.hMul (HMul.hMul (L …
    M'' : Matrix (Fin r) (Fin r) 𝕜 := M'.toBlocks₁₁
    L₀ L₀' : List (Matrix.TransvectionStruct (Fin r) 𝕜)
    D₀ : Fin r → 𝕜
    h₀ : Eq (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L₀) …
    c : 𝕜 := M' (Sum.inr Unit.unit) (Sum.inr Unit.unit)
    this : Eq M' (Matrix.fromBlocks M'' 0 0 (Matrix.diagonal fun x => c))
    ⊢ Eq (HMul.hMul (HMul.hMul (List.map (Function.comp Matrix.TransvectionStruct. …
  -/
  rw [this]
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_3
    inst✝ : Field 𝕜
    r : Nat
    IH : ∀ (M : Matrix (Fin r) (Fin r) 𝕜), Exists fun L₀ => Exists fun L₀' => Exis …
    M : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜
    L₁ L₁' : List (Matrix.TransvectionStruct (Sum (Fin r) Unit) 𝕜)
    hM : (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L₁).pr …
    M' : Matrix (Sum (Fin r) Unit) (Sum (Fin r) Unit) 𝕜 := HMul.hMul (HMul.hMul (L …
    M'' : Matrix (Fin r) (Fin r) 𝕜 := M'.toBlocks₁₁
    L₀ L₀' : List (Matrix.TransvectionStruct (Fin r) 𝕜)
    D₀ : Fin r → 𝕜
    h₀ : Eq (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L₀) …
    c : 𝕜 := M' (Sum.inr Unit.unit) (Sum.inr Unit.unit)
    this : Eq M' (Matrix.fromBlocks M'' 0 0 (Matrix.diagonal fun x => c))
    ⊢ Eq (HMul.hMul (HMul.hMul (List.map (Function.comp Matrix.TransvectionStruct. …
  -/
  simp [h₀]
  /-
    🎉 no goals
  -/


/-- Reduction to diagonal form by elementary operations is invariant under reindexing. -/
theorem reindex_exists_list_transvec_mul_mul_list_transvec_eq_diagonal (M : Matrix p p 𝕜)
    (e : p ≃ n)
    (H :
      ∃ (L L' : List (TransvectionStruct n 𝕜)) (D : n → 𝕜),
        (L.map toMatrix).prod * Matrix.reindexAlgEquiv 𝕜 _ e M * (L'.map toMatrix).prod =
          diagonal D) :
    ∃ (L L' : List (TransvectionStruct p 𝕜)) (D : p → 𝕜),
      (L.map toMatrix).prod * M * (L'.map toMatrix).prod = diagonal D := by
  /-
    n : Type u_1
    p : Type u_2
    𝕜 : Type u_3
    inst✝⁴ : Field 𝕜
    inst✝³ : DecidableEq n
    inst✝² : DecidableEq p
    inst✝¹ : Fintype n
    inst✝ : Fintype p
    M : Matrix p p 𝕜
    e : Equiv p n
    H : Exists fun L => Exists fun L' => Exists fun D => Eq (HMul.hMul (HMul.hMul  …
    ⊢ Exists fun L => Exists fun L' => Exists fun D => Eq (HMul.hMul (HMul.hMul (L …
  -/
  rcases H with ⟨L₀, L₀', D₀, h₀⟩
  /-
    case intro.intro.intro
    n : Type u_1
    p : Type u_2
    𝕜 : Type u_3
    inst✝⁴ : Field 𝕜
    inst✝³ : DecidableEq n
    inst✝² : DecidableEq p
    inst✝¹ : Fintype n
    inst✝ : Fintype p
    M : Matrix p p 𝕜
    e : Equiv p n
    L₀ L₀' : List (Matrix.TransvectionStruct n 𝕜)
    D₀ : n → 𝕜
    h₀ : Eq (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L₀) …
    ⊢ Exists fun L => Exists fun L' => Exists fun D => Eq (HMul.hMul (HMul.hMul (L …
  -/
  refine ⟨L₀.map (reindexEquiv e.symm), L₀'.map (reindexEquiv e.symm), D₀ ∘ e, ?_⟩
  have : M = reindexAlgEquiv 𝕜 _ e.symm (reindexAlgEquiv 𝕜 _ e M) := by
    simp only [Equiv.symm_symm, submatrix_submatrix, reindex_apply, submatrix_id_id,
      Equiv.symm_comp_self, reindexAlgEquiv_apply]
  /-
    case intro.intro.intro
    n : Type u_1
    p : Type u_2
    𝕜 : Type u_3
    inst✝⁴ : Field 𝕜
    inst✝³ : DecidableEq n
    inst✝² : DecidableEq p
    inst✝¹ : Fintype n
    inst✝ : Fintype p
    M : Matrix p p 𝕜
    e : Equiv p n
    L₀ L₀' : List (Matrix.TransvectionStruct n 𝕜)
    D₀ : n → 𝕜
    h₀ : Eq (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L₀) …
    this : Eq M ((Matrix.reindexAlgEquiv 𝕜 𝕜 e.symm) ((Matrix.reindexAlgEquiv 𝕜 𝕜  …
    ⊢ Eq (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix (List. …
  -/
  rw [this]
  /-
    case intro.intro.intro
    n : Type u_1
    p : Type u_2
    𝕜 : Type u_3
    inst✝⁴ : Field 𝕜
    inst✝³ : DecidableEq n
    inst✝² : DecidableEq p
    inst✝¹ : Fintype n
    inst✝ : Fintype p
    M : Matrix p p 𝕜
    e : Equiv p n
    L₀ L₀' : List (Matrix.TransvectionStruct n 𝕜)
    D₀ : n → 𝕜
    h₀ : Eq (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L₀) …
    this : Eq M ((Matrix.reindexAlgEquiv 𝕜 𝕜 e.symm) ((Matrix.reindexAlgEquiv 𝕜 𝕜  …
    ⊢ Eq (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix (List. …
  -/
  simp only [toMatrix_reindexEquiv_prod, List.map_map, reindexAlgEquiv_apply]
  /-
    case intro.intro.intro
    n : Type u_1
    p : Type u_2
    𝕜 : Type u_3
    inst✝⁴ : Field 𝕜
    inst✝³ : DecidableEq n
    inst✝² : DecidableEq p
    inst✝¹ : Fintype n
    inst✝ : Fintype p
    M : Matrix p p 𝕜
    e : Equiv p n
    L₀ L₀' : List (Matrix.TransvectionStruct n 𝕜)
    D₀ : n → 𝕜
    h₀ : Eq (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L₀) …
    this : Eq M ((Matrix.reindexAlgEquiv 𝕜 𝕜 e.symm) ((Matrix.reindexAlgEquiv 𝕜 𝕜  …
    ⊢ Eq (HMul.hMul (HMul.hMul ((Matrix.reindex e.symm e.symm) (List.map Matrix.Tr …
  -/
  simp only [← reindexAlgEquiv_apply 𝕜, ← reindexAlgEquiv_mul, h₀]
  /-
    case intro.intro.intro
    n : Type u_1
    p : Type u_2
    𝕜 : Type u_3
    inst✝⁴ : Field 𝕜
    inst✝³ : DecidableEq n
    inst✝² : DecidableEq p
    inst✝¹ : Fintype n
    inst✝ : Fintype p
    M : Matrix p p 𝕜
    e : Equiv p n
    L₀ L₀' : List (Matrix.TransvectionStruct n 𝕜)
    D₀ : n → 𝕜
    h₀ : Eq (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L₀) …
    this : Eq M ((Matrix.reindexAlgEquiv 𝕜 𝕜 e.symm) ((Matrix.reindexAlgEquiv 𝕜 𝕜  …
    ⊢ Eq ((Matrix.reindexAlgEquiv 𝕜 𝕜 e.symm) (Matrix.diagonal D₀)) (Matrix.diagon …
  -/
  simp only [Equiv.symm_symm, reindex_apply, submatrix_diagonal_equiv, reindexAlgEquiv_apply]
  /-
    🎉 no goals
  -/


/-- Any matrix can be reduced to diagonal form by elementary operations. Formulated here on `Type 0`
because we will make an induction using `Fin r`.
See `exists_list_transvec_mul_mul_list_transvec_eq_diagonal` for the general version (which follows
from this one and reindexing). -/
theorem exists_list_transvec_mul_mul_list_transvec_eq_diagonal_aux (n : Type) [Fintype n]
    [DecidableEq n] (M : Matrix n n 𝕜) :
    ∃ (L L' : List (TransvectionStruct n 𝕜)) (D : n → 𝕜),
      (L.map toMatrix).prod * M * (L'.map toMatrix).prod = diagonal D := by
  /-
    𝕜 : Type u_3
    inst✝² : Field 𝕜
    n : Type
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    M : Matrix n n 𝕜
    ⊢ Exists fun L => Exists fun L' => Exists fun D => Eq (HMul.hMul (HMul.hMul (L …
  -/
  induction' hn : Fintype.card n with r IH generalizing n M
    /-
      case zero
      𝕜 : Type u_3
      inst✝² : Field 𝕜
      n : Type
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      M : Matrix n n 𝕜
      hn : Eq (Fintype.card n) 0
      ⊢ Exists fun L => Exists fun L' => Exists fun D => Eq (HMul.hMul (HMul.hMul (L …
    -/
  · refine ⟨List.nil, List.nil, fun _ => 1, ?_⟩
    /-
      case zero
      𝕜 : Type u_3
      inst✝² : Field 𝕜
      n : Type
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      M : Matrix n n 𝕜
      hn : Eq (Fintype.card n) 0
      ⊢ Eq (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix List.n …
    -/
    ext i j
    /-
      case zero.a
      𝕜 : Type u_3
      inst✝² : Field 𝕜
      n : Type
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      M : Matrix n n 𝕜
      hn : Eq (Fintype.card n) 0
      i j : n
      ⊢ Eq (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix List.n …
    -/
    rw [Fintype.card_eq_zero_iff] at hn
    /-
      case zero.a
      𝕜 : Type u_3
      inst✝² : Field 𝕜
      n : Type
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      M : Matrix n n 𝕜
      hn : IsEmpty n
      i j : n
      ⊢ Eq (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix List.n …
    -/
    exact hn.elim' i
    /-
      🎉 no goals
    -/
  · have e : n ≃ Fin r ⊕ Unit := by
      refine Fintype.equivOfCardEq ?_
      rw [hn]
      rw [@Fintype.card_sum (Fin r) Unit _ _]
      simp
    /-
      case succ
      𝕜 : Type u_3
      inst✝² : Field 𝕜
      r : Nat
      IH : ∀ (n : Type) [inst : Fintype n] [inst_1 : DecidableEq n] (M : Matrix n n  …
      n : Type
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      M : Matrix n n 𝕜
      hn : Eq (Fintype.card n) (HAdd.hAdd r 1)
      e : Equiv n (Sum (Fin r) Unit)
      ⊢ Exists fun L => Exists fun L' => Exists fun D => Eq (HMul.hMul (HMul.hMul (L …
    -/
    apply reindex_exists_list_transvec_mul_mul_list_transvec_eq_diagonal M e
    apply
      exists_list_transvec_mul_mul_list_transvec_eq_diagonal_induction fun N =>
        IH (Fin r) N (by simp)


/-- Any matrix can be reduced to diagonal form by elementary operations. -/
theorem exists_list_transvec_mul_mul_list_transvec_eq_diagonal (M : Matrix n n 𝕜) :
    ∃ (L L' : List (TransvectionStruct n 𝕜)) (D : n → 𝕜),
      (L.map toMatrix).prod * M * (L'.map toMatrix).prod = diagonal D := by
  /-
    n : Type u_1
    𝕜 : Type u_3
    inst✝² : Field 𝕜
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n 𝕜
    ⊢ Exists fun L => Exists fun L' => Exists fun D => Eq (HMul.hMul (HMul.hMul (L …
  -/
  have e : n ≃ Fin (Fintype.card n) := Fintype.equivOfCardEq (by simp)
  /-
    n : Type u_1
    𝕜 : Type u_3
    inst✝² : Field 𝕜
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n 𝕜
    e : Equiv n (Fin (Fintype.card n))
    ⊢ Exists fun L => Exists fun L' => Exists fun D => Eq (HMul.hMul (HMul.hMul (L …
  -/
  apply reindex_exists_list_transvec_mul_mul_list_transvec_eq_diagonal M e
  /-
    n : Type u_1
    𝕜 : Type u_3
    inst✝² : Field 𝕜
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n 𝕜
    e : Equiv n (Fin (Fintype.card n))
    ⊢ Exists fun L => Exists fun L' => Exists fun D => Eq (HMul.hMul (HMul.hMul (L …
  -/
  apply exists_list_transvec_mul_mul_list_transvec_eq_diagonal_aux
  /-
    🎉 no goals
  -/


/-- Any matrix can be written as the product of transvections, a diagonal matrix, and
transvections. -/
theorem exists_list_transvec_mul_diagonal_mul_list_transvec (M : Matrix n n 𝕜) :
    ∃ (L L' : List (TransvectionStruct n 𝕜)) (D : n → 𝕜),
      M = (L.map toMatrix).prod * diagonal D * (L'.map toMatrix).prod := by
  /-
    n : Type u_1
    𝕜 : Type u_3
    inst✝² : Field 𝕜
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n 𝕜
    ⊢ Exists fun L => Exists fun L' => Exists fun D => Eq M (HMul.hMul (HMul.hMul  …
  -/
  rcases exists_list_transvec_mul_mul_list_transvec_eq_diagonal M with ⟨L, L', D, h⟩
  /-
    case intro.intro.intro
    n : Type u_1
    𝕜 : Type u_3
    inst✝² : Field 𝕜
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n 𝕜
    L L' : List (Matrix.TransvectionStruct n 𝕜)
    D : n → 𝕜
    h : Eq (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L).p …
    ⊢ Exists fun L => Exists fun L' => Exists fun D => Eq M (HMul.hMul (HMul.hMul  …
  -/
  refine ⟨L.reverse.map TransvectionStruct.inv, L'.reverse.map TransvectionStruct.inv, D, ?_⟩
  suffices
    M =
      (L.reverse.map (toMatrix ∘ TransvectionStruct.inv)).prod * (L.map toMatrix).prod * M *
        ((L'.map toMatrix).prod * (L'.reverse.map (toMatrix ∘ TransvectionStruct.inv)).prod)
    by simpa [← h, Matrix.mul_assoc]
  /-
    case intro.intro.intro
    n : Type u_1
    𝕜 : Type u_3
    inst✝² : Field 𝕜
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n 𝕜
    L L' : List (Matrix.TransvectionStruct n 𝕜)
    D : n → 𝕜
    h : Eq (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L).p …
    ⊢ Eq M (HMul.hMul (HMul.hMul (HMul.hMul (List.map (Function.comp Matrix.Transv …
  -/
  rw [reverse_inv_prod_mul_prod, prod_mul_reverse_inv_prod, Matrix.one_mul, Matrix.mul_one]
  /-
    🎉 no goals
  -/


/-- Induction principle for matrices based on transvections: if a property is true for all diagonal
matrices, all transvections, and is stable under product, then it is true for all matrices. This is
the useful way to say that matrices are generated by diagonal matrices and transvections.

We state a slightly more general version: to prove a property for a matrix `M`, it suffices to
assume that the diagonal matrices we consider have the same determinant as `M`. This is useful to
obtain similar principles for `SLₙ` or `GLₙ`. -/
theorem diagonal_transvection_induction (P : Matrix n n 𝕜 → Prop) (M : Matrix n n 𝕜)
    (hdiag : ∀ D : n → 𝕜, det (diagonal D) = det M → P (diagonal D))
    (htransvec : ∀ t : TransvectionStruct n 𝕜, P t.toMatrix) (hmul : ∀ A B, P A → P B → P (A * B)) :
    P M := by
  /-
    n : Type u_1
    𝕜 : Type u_3
    inst✝² : Field 𝕜
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    P : Matrix n n 𝕜 → Prop
    M : Matrix n n 𝕜
    hdiag : ∀ (D : n → 𝕜), Eq (Matrix.diagonal D).det M.det → P (Matrix.diagonal D)
    htransvec : ∀ (t : Matrix.TransvectionStruct n 𝕜), P t.toMatrix
    hmul : ∀ (A B : Matrix n n 𝕜), P A → P B → P (HMul.hMul A B)
    ⊢ P M
  -/
  rcases exists_list_transvec_mul_diagonal_mul_list_transvec M with ⟨L, L', D, h⟩
  /-
    case intro.intro.intro
    n : Type u_1
    𝕜 : Type u_3
    inst✝² : Field 𝕜
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    P : Matrix n n 𝕜 → Prop
    M : Matrix n n 𝕜
    hdiag : ∀ (D : n → 𝕜), Eq (Matrix.diagonal D).det M.det → P (Matrix.diagonal D)
    htransvec : ∀ (t : Matrix.TransvectionStruct n 𝕜), P t.toMatrix
    hmul : ∀ (A B : Matrix n n 𝕜), P A → P B → P (HMul.hMul A B)
    L L' : List (Matrix.TransvectionStruct n 𝕜)
    D : n → 𝕜
    h : Eq M (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L) …
    ⊢ P M
  -/
  have PD : P (diagonal D) := hdiag D (by simp [h])
  suffices H :
    ∀ (L₁ L₂ : List (TransvectionStruct n 𝕜)) (E : Matrix n n 𝕜),
      P E → P ((L₁.map toMatrix).prod * E * (L₂.map toMatrix).prod) by
    rw [h]
    apply H L L'
    exact PD
  /-
    case intro.intro.intro
    n : Type u_1
    𝕜 : Type u_3
    inst✝² : Field 𝕜
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    P : Matrix n n 𝕜 → Prop
    M : Matrix n n 𝕜
    hdiag : ∀ (D : n → 𝕜), Eq (Matrix.diagonal D).det M.det → P (Matrix.diagonal D)
    htransvec : ∀ (t : Matrix.TransvectionStruct n 𝕜), P t.toMatrix
    hmul : ∀ (A B : Matrix n n 𝕜), P A → P B → P (HMul.hMul A B)
    L L' : List (Matrix.TransvectionStruct n 𝕜)
    D : n → 𝕜
    h : Eq M (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L) …
    PD : P (Matrix.diagonal D)
    ⊢ ∀ (L₁ L₂ : List (Matrix.TransvectionStruct n 𝕜)) (E : Matrix n n 𝕜), P E → P …
  -/
  intro L₁ L₂ E PE
  /-
    case intro.intro.intro
    n : Type u_1
    𝕜 : Type u_3
    inst✝² : Field 𝕜
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    P : Matrix n n 𝕜 → Prop
    M : Matrix n n 𝕜
    hdiag : ∀ (D : n → 𝕜), Eq (Matrix.diagonal D).det M.det → P (Matrix.diagonal D)
    htransvec : ∀ (t : Matrix.TransvectionStruct n 𝕜), P t.toMatrix
    hmul : ∀ (A B : Matrix n n 𝕜), P A → P B → P (HMul.hMul A B)
    L L' : List (Matrix.TransvectionStruct n 𝕜)
    D : n → 𝕜
    h : Eq M (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L) …
    PD : P (Matrix.diagonal D)
    L₁ L₂ : List (Matrix.TransvectionStruct n 𝕜)
    E : Matrix n n 𝕜
    PE : P E
    ⊢ P (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L₁).pro …
  -/
  induction' L₁ with t L₁ IH
    /-
      case intro.intro.intro.nil
      n : Type u_1
      𝕜 : Type u_3
      inst✝² : Field 𝕜
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      P : Matrix n n 𝕜 → Prop
      M : Matrix n n 𝕜
      hdiag : ∀ (D : n → 𝕜), Eq (Matrix.diagonal D).det M.det → P (Matrix.diagonal D)
      htransvec : ∀ (t : Matrix.TransvectionStruct n 𝕜), P t.toMatrix
      hmul : ∀ (A B : Matrix n n 𝕜), P A → P B → P (HMul.hMul A B)
      L L' : List (Matrix.TransvectionStruct n 𝕜)
      D : n → 𝕜
      h : Eq M (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L) …
      PD : P (Matrix.diagonal D)
      L₂ : List (Matrix.TransvectionStruct n 𝕜)
      E : Matrix n n 𝕜
      PE : P E
      ⊢ P (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix List.ni …
    -/
  · simp only [Matrix.one_mul, List.prod_nil, List.map]
    /-
      case intro.intro.intro.nil
      n : Type u_1
      𝕜 : Type u_3
      inst✝² : Field 𝕜
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      P : Matrix n n 𝕜 → Prop
      M : Matrix n n 𝕜
      hdiag : ∀ (D : n → 𝕜), Eq (Matrix.diagonal D).det M.det → P (Matrix.diagonal D)
      htransvec : ∀ (t : Matrix.TransvectionStruct n 𝕜), P t.toMatrix
      hmul : ∀ (A B : Matrix n n 𝕜), P A → P B → P (HMul.hMul A B)
      L L' : List (Matrix.TransvectionStruct n 𝕜)
      D : n → 𝕜
      h : Eq M (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L) …
      PD : P (Matrix.diagonal D)
      L₂ : List (Matrix.TransvectionStruct n 𝕜)
      E : Matrix n n 𝕜
      PE : P E
      ⊢ P (HMul.hMul E (List.map Matrix.TransvectionStruct.toMatrix L₂).prod)
    -/
    induction' L₂ with t L₂ IH generalizing E
      /-
        case intro.intro.intro.nil.nil
        n : Type u_1
        𝕜 : Type u_3
        inst✝² : Field 𝕜
        inst✝¹ : DecidableEq n
        inst✝ : Fintype n
        P : Matrix n n 𝕜 → Prop
        M : Matrix n n 𝕜
        hdiag : ∀ (D : n → 𝕜), Eq (Matrix.diagonal D).det M.det → P (Matrix.diagonal D)
        htransvec : ∀ (t : Matrix.TransvectionStruct n 𝕜), P t.toMatrix
        hmul : ∀ (A B : Matrix n n 𝕜), P A → P B → P (HMul.hMul A B)
        L L' : List (Matrix.TransvectionStruct n 𝕜)
        D : n → 𝕜
        h : Eq M (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L) …
        PD : P (Matrix.diagonal D)
        E : Matrix n n 𝕜
        PE : P E
        ⊢ P (HMul.hMul E (List.map Matrix.TransvectionStruct.toMatrix List.nil).prod)
      -/
    · simpa
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.nil.cons
        n : Type u_1
        𝕜 : Type u_3
        inst✝² : Field 𝕜
        inst✝¹ : DecidableEq n
        inst✝ : Fintype n
        P : Matrix n n 𝕜 → Prop
        M : Matrix n n 𝕜
        hdiag : ∀ (D : n → 𝕜), Eq (Matrix.diagonal D).det M.det → P (Matrix.diagonal D)
        htransvec : ∀ (t : Matrix.TransvectionStruct n 𝕜), P t.toMatrix
        hmul : ∀ (A B : Matrix n n 𝕜), P A → P B → P (HMul.hMul A B)
        L L' : List (Matrix.TransvectionStruct n 𝕜)
        D : n → 𝕜
        h : Eq M (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L) …
        PD : P (Matrix.diagonal D)
        t : Matrix.TransvectionStruct n 𝕜
        L₂ : List (Matrix.TransvectionStruct n 𝕜)
        IH : ∀ (E : Matrix n n 𝕜), P E → P (HMul.hMul E (List.map Matrix.TransvectionS …
        E : Matrix n n 𝕜
        PE : P E
        ⊢ P (HMul.hMul E (List.map Matrix.TransvectionStruct.toMatrix (List.cons t L₂) …
      -/
    · simp only [← Matrix.mul_assoc, List.prod_cons, List.map]
      /-
        case intro.intro.intro.nil.cons
        n : Type u_1
        𝕜 : Type u_3
        inst✝² : Field 𝕜
        inst✝¹ : DecidableEq n
        inst✝ : Fintype n
        P : Matrix n n 𝕜 → Prop
        M : Matrix n n 𝕜
        hdiag : ∀ (D : n → 𝕜), Eq (Matrix.diagonal D).det M.det → P (Matrix.diagonal D)
        htransvec : ∀ (t : Matrix.TransvectionStruct n 𝕜), P t.toMatrix
        hmul : ∀ (A B : Matrix n n 𝕜), P A → P B → P (HMul.hMul A B)
        L L' : List (Matrix.TransvectionStruct n 𝕜)
        D : n → 𝕜
        h : Eq M (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L) …
        PD : P (Matrix.diagonal D)
        t : Matrix.TransvectionStruct n 𝕜
        L₂ : List (Matrix.TransvectionStruct n 𝕜)
        IH : ∀ (E : Matrix n n 𝕜), P E → P (HMul.hMul E (List.map Matrix.TransvectionS …
        E : Matrix n n 𝕜
        PE : P E
        ⊢ P (HMul.hMul (HMul.hMul E t.toMatrix) (List.map Matrix.TransvectionStruct.to …
      -/
      apply IH
      /-
        case intro.intro.intro.nil.cons.PE
        n : Type u_1
        𝕜 : Type u_3
        inst✝² : Field 𝕜
        inst✝¹ : DecidableEq n
        inst✝ : Fintype n
        P : Matrix n n 𝕜 → Prop
        M : Matrix n n 𝕜
        hdiag : ∀ (D : n → 𝕜), Eq (Matrix.diagonal D).det M.det → P (Matrix.diagonal D)
        htransvec : ∀ (t : Matrix.TransvectionStruct n 𝕜), P t.toMatrix
        hmul : ∀ (A B : Matrix n n 𝕜), P A → P B → P (HMul.hMul A B)
        L L' : List (Matrix.TransvectionStruct n 𝕜)
        D : n → 𝕜
        h : Eq M (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L) …
        PD : P (Matrix.diagonal D)
        t : Matrix.TransvectionStruct n 𝕜
        L₂ : List (Matrix.TransvectionStruct n 𝕜)
        IH : ∀ (E : Matrix n n 𝕜), P E → P (HMul.hMul E (List.map Matrix.TransvectionS …
        E : Matrix n n 𝕜
        PE : P E
        ⊢ P (HMul.hMul E t.toMatrix)
      -/
      exact hmul _ _ PE (htransvec _)
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.intro.cons
      n : Type u_1
      𝕜 : Type u_3
      inst✝² : Field 𝕜
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      P : Matrix n n 𝕜 → Prop
      M : Matrix n n 𝕜
      hdiag : ∀ (D : n → 𝕜), Eq (Matrix.diagonal D).det M.det → P (Matrix.diagonal D)
      htransvec : ∀ (t : Matrix.TransvectionStruct n 𝕜), P t.toMatrix
      hmul : ∀ (A B : Matrix n n 𝕜), P A → P B → P (HMul.hMul A B)
      L L' : List (Matrix.TransvectionStruct n 𝕜)
      D : n → 𝕜
      h : Eq M (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L) …
      PD : P (Matrix.diagonal D)
      L₂ : List (Matrix.TransvectionStruct n 𝕜)
      E : Matrix n n 𝕜
      PE : P E
      t : Matrix.TransvectionStruct n 𝕜
      L₁ : List (Matrix.TransvectionStruct n 𝕜)
      IH : P (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L₁). …
      ⊢ P (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix (List.c …
    -/
  · simp only [Matrix.mul_assoc, List.prod_cons, List.map] at IH ⊢
    /-
      case intro.intro.intro.cons
      n : Type u_1
      𝕜 : Type u_3
      inst✝² : Field 𝕜
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      P : Matrix n n 𝕜 → Prop
      M : Matrix n n 𝕜
      hdiag : ∀ (D : n → 𝕜), Eq (Matrix.diagonal D).det M.det → P (Matrix.diagonal D)
      htransvec : ∀ (t : Matrix.TransvectionStruct n 𝕜), P t.toMatrix
      hmul : ∀ (A B : Matrix n n 𝕜), P A → P B → P (HMul.hMul A B)
      L L' : List (Matrix.TransvectionStruct n 𝕜)
      D : n → 𝕜
      h : Eq M (HMul.hMul (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L) …
      PD : P (Matrix.diagonal D)
      L₂ : List (Matrix.TransvectionStruct n 𝕜)
      E : Matrix n n 𝕜
      PE : P E
      t : Matrix.TransvectionStruct n 𝕜
      L₁ : List (Matrix.TransvectionStruct n 𝕜)
      IH : P (HMul.hMul (List.map Matrix.TransvectionStruct.toMatrix L₁).prod (HMul. …
      ⊢ P (HMul.hMul t.toMatrix (HMul.hMul (List.map Matrix.TransvectionStruct.toMat …
    -/
    exact hmul _ _ (htransvec _) IH
    /-
      🎉 no goals
    -/


/-- Induction principle for invertible matrices based on transvections: if a property is true for
all invertible diagonal matrices, all transvections, and is stable under product of invertible
matrices, then it is true for all invertible matrices. This is the useful way to say that
invertible matrices are generated by invertible diagonal matrices and transvections. -/
theorem diagonal_transvection_induction_of_det_ne_zero (P : Matrix n n 𝕜 → Prop) (M : Matrix n n 𝕜)
    (hMdet : det M ≠ 0) (hdiag : ∀ D : n → 𝕜, det (diagonal D) ≠ 0 → P (diagonal D))
    (htransvec : ∀ t : TransvectionStruct n 𝕜, P t.toMatrix)
    (hmul : ∀ A B, det A ≠ 0 → det B ≠ 0 → P A → P B → P (A * B)) : P M := by
  /-
    n : Type u_1
    𝕜 : Type u_3
    inst✝² : Field 𝕜
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    P : Matrix n n 𝕜 → Prop
    M : Matrix n n 𝕜
    hMdet : Ne M.det 0
    hdiag : ∀ (D : n → 𝕜), Ne (Matrix.diagonal D).det 0 → P (Matrix.diagonal D)
    htransvec : ∀ (t : Matrix.TransvectionStruct n 𝕜), P t.toMatrix
    hmul : ∀ (A B : Matrix n n 𝕜), Ne A.det 0 → Ne B.det 0 → P A → P B → P (HMul.h …
    ⊢ P M
  -/
  let Q : Matrix n n 𝕜 → Prop := fun N => det N ≠ 0 ∧ P N
  have : Q M := by
    apply diagonal_transvection_induction Q M
    · intro D hD
      have detD : det (diagonal D) ≠ 0 := by
        rw [hD]
        exact hMdet
      exact ⟨detD, hdiag _ detD⟩
    · intro t
      exact ⟨by simp, htransvec t⟩
    · intro A B QA QB
      exact ⟨by simp [QA.1, QB.1], hmul A B QA.1 QB.1 QA.2 QB.2⟩
  /-
    n : Type u_1
    𝕜 : Type u_3
    inst✝² : Field 𝕜
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    P : Matrix n n 𝕜 → Prop
    M : Matrix n n 𝕜
    hMdet : Ne M.det 0
    hdiag : ∀ (D : n → 𝕜), Ne (Matrix.diagonal D).det 0 → P (Matrix.diagonal D)
    htransvec : ∀ (t : Matrix.TransvectionStruct n 𝕜), P t.toMatrix
    hmul : ∀ (A B : Matrix n n 𝕜), Ne A.det 0 → Ne B.det 0 → P A → P B → P (HMul.h …
    Q : Matrix n n 𝕜 → Prop := fun N => And (Ne N.det 0) (P N)
    this : Q M
    ⊢ P M
  -/
  exact this.2
  /-
    🎉 no goals
  -/


