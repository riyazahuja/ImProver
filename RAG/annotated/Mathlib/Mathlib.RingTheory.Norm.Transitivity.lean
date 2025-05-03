/-- Given a ((m-1)+1)x((m-1)+1) block matrix `M = [[A,b],[c,d]]`, `auxMat M k` is the auxiliary
matrix `[[dI,0],[-c,1]]`. `k` corresponds to the last row/column of the matrix. -/
def auxMat : Matrix m m S :=
  of fun i j ↦
    if j = k then
      if i = k then 1 else 0
    else if i = k then -M k j
    else if i = j then M k k
    else 0


/-- `aux M k` is lower triangular. -/
lemma auxMat_blockTriangular : (auxMat M k).BlockTriangular (· ≠ k) :=
  fun i j lt ↦ by
    /-
      S : Type u_2
      m : Type u_5
      inst✝¹ : CommRing S
      M : Matrix m m S
      inst✝ : DecidableEq m
      k i j : m
      lt : LT.lt ((fun x => Ne x k) j) ((fun x => Ne x k) i)
      ⊢ Eq (Algebra.Norm.Transitivity.auxMat M k i j) 0
    -/
    simp_rw [lt_iff_not_le, le_Prop_eq, Classical.not_imp, not_not] at lt
    /-
      S : Type u_2
      m : Type u_5
      inst✝¹ : CommRing S
      M : Matrix m m S
      inst✝ : DecidableEq m
      k i j : m
      lt : And (Ne i k) (Eq j k)
      ⊢ Eq (Algebra.Norm.Transitivity.auxMat M k i j) 0
    -/
    rw [auxMat, of_apply, if_pos lt.2, if_neg lt.1]
    /-
      🎉 no goals
    -/


lemma auxMat_toSquareBlock_ne : (auxMat M k).toSquareBlock (· ≠ k) True = M k k • 1 := by
  /-
    S : Type u_2
    m : Type u_5
    inst✝¹ : CommRing S
    M : Matrix m m S
    inst✝ : DecidableEq m
    k : m
    ⊢ Eq ((Algebra.Norm.Transitivity.auxMat M k).toSquareBlock (fun x => Ne x k) T …
  -/
  ext i j
  simp [auxMat, toSquareBlock_def, if_neg (of_eq_true i.2), if_neg (of_eq_true j.2),
    Matrix.one_apply, Subtype.ext_iff]


lemma auxMat_toSquareBlock_eq : (auxMat M k).toSquareBlock (· ≠ k) False = 1 := by
  /-
    S : Type u_2
    m : Type u_5
    inst✝¹ : CommRing S
    M : Matrix m m S
    inst✝ : DecidableEq m
    k : m
    ⊢ Eq ((Algebra.Norm.Transitivity.auxMat M k).toSquareBlock (fun x => Ne x k) F …
  -/
  ext ⟨i, hi⟩ ⟨j, hj⟩
  /-
    case a.mk.mk
    S : Type u_2
    m : Type u_5
    inst✝¹ : CommRing S
    M : Matrix m m S
    inst✝ : DecidableEq m
    k i : m
    hi : Eq (Ne i k) False
    j : m
    hj : Eq (Ne j k) False
    ⊢ Eq ((Algebra.Norm.Transitivity.auxMat M k).toSquareBlock (fun x => Ne x k) F …
  -/
  rw [eq_iff_iff, iff_false, not_not] at hi hj
  /-
    case a.mk.mk
    S : Type u_2
    m : Type u_5
    inst✝¹ : CommRing S
    M : Matrix m m S
    inst✝ : DecidableEq m
    k i : m
    hi✝ : Eq (Ne i k) False
    hi : Eq i k
    j : m
    hj✝ : Eq (Ne j k) False
    hj : Eq j k
    ⊢ Eq ((Algebra.Norm.Transitivity.auxMat M k).toSquareBlock (fun x => Ne x k) F …
  -/
  simp [auxMat, toSquareBlock_def, if_pos hi, if_pos hj, Matrix.one_apply, if_pos (hj ▸ hi)]
  /-
    🎉 no goals
  -/


/-- `M * aux M k` is upper triangular. -/
lemma mul_auxMat_blockTriangular : (M * auxMat M k).BlockTriangular (· = k) :=
  fun i j lt ↦ by
    /-
      S : Type u_2
      m : Type u_5
      inst✝² : CommRing S
      M : Matrix m m S
      inst✝¹ : DecidableEq m
      k : m
      inst✝ : Fintype m
      i j : m
      lt : LT.lt ((fun x => Eq x k) j) ((fun x => Eq x k) i)
      ⊢ Eq (HMul.hMul M (Algebra.Norm.Transitivity.auxMat M k) i j) 0
    -/
    simp_rw [lt_iff_not_le, le_Prop_eq, Classical.not_imp] at lt
    /-
      S : Type u_2
      m : Type u_5
      inst✝² : CommRing S
      M : Matrix m m S
      inst✝¹ : DecidableEq m
      k : m
      inst✝ : Fintype m
      i j : m
      lt : And (Eq i k) (Not (Eq j k))
      ⊢ Eq (HMul.hMul M (Algebra.Norm.Transitivity.auxMat M k) i j) 0
    -/
    simp_rw [Matrix.mul_apply, auxMat, of_apply, if_neg lt.2, mul_ite, mul_neg, mul_zero]
    rw [Finset.sum_ite, Finset.filter_eq', if_pos (Finset.mem_univ _), Finset.sum_singleton,
      Finset.sum_ite_eq', if_pos, lt.1, mul_comm, neg_add_cancel]
    /-
      case hc
      S : Type u_2
      m : Type u_5
      inst✝² : CommRing S
      M : Matrix m m S
      inst✝¹ : DecidableEq m
      k : m
      inst✝ : Fintype m
      i j : m
      lt : And (Eq i k) (Not (Eq j k))
      ⊢ Membership.mem (Finset.filter (fun x => Not (Eq x k)) Finset.univ) j
    -/
    exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, lt.2⟩
    /-
      🎉 no goals
    -/


/-- The lower-right corner of `M * aux M k` is the same as the corner of `M`. -/
                                                             /-
                                                               S : Type u_2
                                                               m : Type u_5
                                                               inst✝² : CommRing S
                                                               M : Matrix m m S
                                                               inst✝¹ : DecidableEq m
                                                               k : m
                                                               inst✝ : Fintype m
                                                               ⊢ Eq (HMul.hMul M (Algebra.Norm.Transitivity.auxMat M k) k k) (M k k)
                                                             -/
lemma mul_auxMat_corner : (M * auxMat M k) k k = M k k := by simp [Matrix.mul_apply, auxMat]
                                                             /-
                                                               🎉 no goals
                                                             -/


lemma mul_auxMat_toSquareBlock_eq :
    (M * auxMat M k).toSquareBlock (· = k) True = M k k • 1 := by
  /-
    S : Type u_2
    m : Type u_5
    inst✝² : CommRing S
    M : Matrix m m S
    inst✝¹ : DecidableEq m
    k : m
    inst✝ : Fintype m
    ⊢ Eq ((HMul.hMul M (Algebra.Norm.Transitivity.auxMat M k)).toSquareBlock (fun  …
  -/
  ext ⟨i, hi⟩ ⟨j, hj⟩
  /-
    case a.mk.mk
    S : Type u_2
    m : Type u_5
    inst✝² : CommRing S
    M : Matrix m m S
    inst✝¹ : DecidableEq m
    k : m
    inst✝ : Fintype m
    i : m
    hi : Eq (Eq i k) True
    j : m
    hj : Eq (Eq j k) True
    ⊢ Eq ((HMul.hMul M (Algebra.Norm.Transitivity.auxMat M k)).toSquareBlock (fun  …
  -/
  rw [eq_iff_iff, iff_true] at hi hj
  /-
    case a.mk.mk
    S : Type u_2
    m : Type u_5
    inst✝² : CommRing S
    M : Matrix m m S
    inst✝¹ : DecidableEq m
    k : m
    inst✝ : Fintype m
    i : m
    hi✝ : Eq (Eq i k) True
    hi : Eq i k
    j : m
    hj✝ : Eq (Eq j k) True
    hj : Eq j k
    ⊢ Eq ((HMul.hMul M (Algebra.Norm.Transitivity.auxMat M k)).toSquareBlock (fun  …
  -/
  simp [toSquareBlock_def, hi, hj, mul_auxMat_corner]
  /-
    🎉 no goals
  -/


set_option quotPrecheck false in
/-- The upper-left block of `M * aux M k`. -/
scoped notation "mulAuxMatBlock" => (M * auxMat M k).toSquareBlock (· = k) False


lemma det_mul_corner_pow :
    M.det * M k k ^ (Fintype.card m - 1) = M k k * (mulAuxMatBlock).det := by
  /-
    S : Type u_2
    m : Type u_5
    inst✝² : CommRing S
    M : Matrix m m S
    inst✝¹ : DecidableEq m
    k : m
    inst✝ : Fintype m
    ⊢ Eq (HMul.hMul M.det (HPow.hPow (M k k) (HSub.hSub (Fintype.card m) 1))) (HMu …
  -/
  trans (M * auxMat M k).det
  · simp [det_mul, (auxMat_blockTriangular M k).det_fintype,
      auxMat_toSquareBlock_ne, auxMat_toSquareBlock_eq]
  /-
    S : Type u_2
    m : Type u_5
    inst✝² : CommRing S
    M : Matrix m m S
    inst✝¹ : DecidableEq m
    k : m
    inst✝ : Fintype m
    ⊢ Eq (HMul.hMul M (Algebra.Norm.Transitivity.auxMat M k)).det (HMul.hMul (M k  …
  -/
  rw [(mul_auxMat_blockTriangular M k).det_fintype, Fintype.prod_Prop, mul_auxMat_toSquareBlock_eq]
  simp_rw [det_smul_of_tower, eq_iff_iff, iff_true, Fintype.card_unique,
    pow_one, det_one, smul_eq_mul, mul_one]
  -- `Decidable (P = Q)` diamond induced by `Prop.linearOrder`, which is classical, when `P` and `Q`
  -- are themselves decidable.
  /-
    S : Type u_2
    m : Type u_5
    inst✝² : CommRing S
    M : Matrix m m S
    inst✝¹ : DecidableEq m
    k : m
    inst✝ : Fintype m
    ⊢ Eq (HMul.hMul (M k k) ((HMul.hMul M (Algebra.Norm.Transitivity.auxMat M k)). …
  -/
  convert rfl
  /-
    🎉 no goals
  -/


/-- A matrix with X added to the corner. -/
noncomputable def cornerAddX : Matrix m m S[X] :=
  (diagonal fun i ↦ if i = k then X else 0) + M.map C


omit [Fintype m] in
lemma polyToMatrix_cornerAddX :
    f.polyToMatrix (cornerAddX M k k k) = (-f (M k k)).charmatrix := by
  simp [cornerAddX, Matrix.add_apply, charmatrix,
    RingHom.polyToMatrix, ← AlgEquiv.symm_toRingEquiv, map_neg]


lemma eval_zero_det_det : eval 0 (f.polyToMatrix (cornerAddX M k).det).det = (f M.det).det := by
  rw [← coe_evalRingHom, RingHom.map_det, ← RingHom.comp_apply,
    evalRingHom_mapMatrix_comp_polyToMatrix, f.comp_apply, RingHom.map_det]
  /-
    R : Type u_1
    S : Type u_2
    n : Type u_4
    m : Type u_5
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    M : Matrix m m S
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    k : m
    inst✝¹ : Fintype m
    inst✝ : Fintype n
    f : RingHom S (Matrix n n R)
    ⊢ Eq (f ((Polynomial.evalRingHom 0).mapMatrix (Algebra.Norm.Transitivity.corne …
  -/
  congr; ext; simp [cornerAddX, diagonal, apply_ite]
              /-
                🎉 no goals
              -/


lemma eval_zero_comp_det :
    eval 0 (comp m m n n R[X] <| (cornerAddX M k).map f.polyToMatrix).det =
      (comp m m n n R <| M.map f).det := by
  simp_rw [← coe_evalRingHom, RingHom.map_det, ← compRingEquiv_apply, ← RingEquiv.coe_toRingHom,
    ← RingHom.mapMatrix_apply, ← RingHom.comp_apply, ← RingHom.comp_assoc,
    evalRingHom_mapMatrix_comp_compRingEquiv, RingHom.comp_assoc, RingHom.mapMatrix_comp,
    evalRingHom_mapMatrix_comp_polyToMatrix, ← RingHom.mapMatrix_comp, RingHom.comp_apply]
  /-
    R : Type u_1
    S : Type u_2
    n : Type u_4
    m : Type u_5
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    M : Matrix m m S
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    k : m
    inst✝¹ : Fintype m
    inst✝ : Fintype n
    f : RingHom S (Matrix n n R)
    ⊢ Eq ((Matrix.compRingEquiv m n R).toRingHom (f.mapMatrix ((Polynomial.evalRin …
  -/
  congr with i j
  /-
    case e_M.h.e_6.h.h.e_6.h.a
    R : Type u_1
    S : Type u_2
    n : Type u_4
    m : Type u_5
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    M : Matrix m m S
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    k : m
    inst✝¹ : Fintype m
    inst✝ : Fintype n
    f : RingHom S (Matrix n n R)
    i j : m
    ⊢ Eq ((Polynomial.evalRingHom 0).mapMatrix (Algebra.Norm.Transitivity.cornerAd …
  -/
  simp [cornerAddX, diagonal, apply_ite]
  /-
    🎉 no goals
  -/


theorem comp_det_mul_pow :
    ((M.map f).comp m m n n R).det * (f (M k k)).det ^ (Fintype.card m - 1) =
      (f (M k k)).det * (((mulAuxMatBlock).map f).comp _ _ n n R).det := by
  /-
    R : Type u_1
    S : Type u_2
    n : Type u_4
    m : Type u_5
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    M : Matrix m m S
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    k : m
    inst✝¹ : Fintype m
    inst✝ : Fintype n
    f : RingHom S (Matrix n n R)
    ⊢ Eq (HMul.hMul ((Matrix.comp m m n n R) (M.map ⇑f)).det (HPow.hPow (f (M k k) …
  -/
  trans (((M * auxMat M k).map f).comp m m n n R).det
  · simp_rw [← f.mapMatrix_apply, ← compRingEquiv_apply, _root_.map_mul, det_mul, f.mapMatrix_apply,
      compRingEquiv_apply, ((auxMat_blockTriangular M k).map f).comp.det_fintype, Fintype.prod_Prop,
      comp_toSquareBlock (b := (· ≠ k)), det_reindex_self, map_toSquareBlock,
      auxMat_toSquareBlock_eq, auxMat_toSquareBlock_ne, smul_one_eq_diagonal, ← diagonal_one,
      diagonal_map (map_zero _), comp_diagonal, det_reindex_self]
    /-
      R : Type u_1
      S : Type u_2
      n : Type u_4
      m : Type u_5
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing S
      M : Matrix m m S
      inst✝³ : DecidableEq m
      inst✝² : DecidableEq n
      k : m
      inst✝¹ : Fintype m
      inst✝ : Fintype n
      f : RingHom S (Matrix n n R)
      ⊢ Eq (HMul.hMul ((Matrix.comp m m n n R) (M.map ⇑f)).det (HPow.hPow (f (M k k) …
    -/
    simp
    /-
      🎉 no goals
    -/
  · simp_rw [((mul_auxMat_blockTriangular M k).map f).comp.det_fintype, Fintype.prod_Prop,
      comp_toSquareBlock (b := (· = k)), det_reindex_self, map_toSquareBlock,
      mul_auxMat_toSquareBlock_eq, smul_one_eq_diagonal,
      diagonal_map (map_zero _), comp_diagonal, det_reindex_self]
    /-
      R : Type u_1
      S : Type u_2
      n : Type u_4
      m : Type u_5
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing S
      M : Matrix m m S
      inst✝³ : DecidableEq m
      inst✝² : DecidableEq n
      k : m
      inst✝¹ : Fintype m
      inst✝ : Fintype n
      f : RingHom S (Matrix n n R)
      ⊢ Eq (HMul.hMul (Matrix.blockDiagonal fun m => f (M k k)).det ((Matrix.comp (S …
    -/
    simp
    /-
      🎉 no goals
    -/


variable {M f} in
lemma det_det_aux
    (ih : ∀ M, (f (det M)).det = ((M.map f).comp {a // (a = k) = False} _ n n R).det) :
    ((f M.det).det - ((M.map f).comp m m n n R).det) *
      (f (M k k)).det ^ (Fintype.card m - 1) = 0 := by
  rw [sub_mul, comp_det_mul_pow, ← det_pow, ← map_pow, ← det_mul, ← _root_.map_mul,
    det_mul_corner_pow, _root_.map_mul, det_mul, ih, sub_self]


/-- The main result in Silvester's paper *Determinants of Block Matrices*: the determinant of
a block matrix with commuting, equal-sized, square blocks can be computed by taking determinants
twice in a row: first take the determinant over the commutative ring generated by the
blocks (`S` here), then take the determinant over the base ring. -/
theorem Matrix.det_det [Fintype m] [Fintype n] (f : S →+* Matrix n n R) :
    (f M.det).det = ((M.map f).comp m m n n R).det := by
  /-
    R : Type u_1
    S : Type u_2
    n : Type u_4
    m : Type u_5
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    M : Matrix m m S
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    inst✝¹ : Fintype m
    inst✝ : Fintype n
    f : RingHom S (Matrix n n R)
    ⊢ Eq (f M.det).det ((Matrix.comp m m n n R) (M.map ⇑f)).det
  -/
  set l := Fintype.card m with hl
  /-
    R : Type u_1
    S : Type u_2
    n : Type u_4
    m : Type u_5
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    M : Matrix m m S
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    inst✝¹ : Fintype m
    inst✝ : Fintype n
    f : RingHom S (Matrix n n R)
    l : Nat := Fintype.card m
    hl : Eq l (Fintype.card m)
    ⊢ Eq (f M.det).det ((Matrix.comp m m n n R) (M.map ⇑f)).det
  -/
  clear_value l; revert R S m
  /-
    n : Type u_4
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    l : Nat
    ⊢ ∀ {R : Type u_1} {S : Type u_2} {m : Type u_5} [inst : CommRing R] [inst_1 : …
  -/
  induction' l with l ih <;> intro R S m _ _ M _ _ f card
    /-
      case zero
      n : Type u_4
      inst✝⁵ : DecidableEq n
      inst✝⁴ : Fintype n
      R : Type u_1
      S : Type u_2
      m : Type u_5
      inst✝³ : CommRing R
      inst✝² : CommRing S
      M : Matrix m m S
      inst✝¹ : DecidableEq m
      inst✝ : Fintype m
      f : RingHom S (Matrix n n R)
      card : Eq 0 (Fintype.card m)
      ⊢ Eq (f M.det).det ((Matrix.comp m m n n R) (M.map ⇑f)).det
    -/
  · rw [eq_comm, Fintype.card_eq_zero_iff] at card
    /-
      case zero
      n : Type u_4
      inst✝⁵ : DecidableEq n
      inst✝⁴ : Fintype n
      R : Type u_1
      S : Type u_2
      m : Type u_5
      inst✝³ : CommRing R
      inst✝² : CommRing S
      M : Matrix m m S
      inst✝¹ : DecidableEq m
      inst✝ : Fintype m
      f : RingHom S (Matrix n n R)
      card : IsEmpty m
      ⊢ Eq (f M.det).det ((Matrix.comp m m n n R) (M.map ⇑f)).det
    -/
    simp_rw [Matrix.det_isEmpty, _root_.map_one, det_one]
    /-
      🎉 no goals
    -/
  /-
    case succ
    n : Type u_4
    inst✝⁵ : DecidableEq n
    inst✝⁴ : Fintype n
    l : Nat
    ih : ∀ {R : Type u_1} {S : Type u_2} {m : Type u_5} [inst : CommRing R] [inst_ …
    R : Type u_1
    S : Type u_2
    m : Type u_5
    inst✝³ : CommRing R
    inst✝² : CommRing S
    M : Matrix m m S
    inst✝¹ : DecidableEq m
    inst✝ : Fintype m
    f : RingHom S (Matrix n n R)
    card : Eq (HAdd.hAdd l 1) (Fintype.card m)
    ⊢ Eq (f M.det).det ((Matrix.comp m m n n R) (M.map ⇑f)).det
  -/
  have ⟨k⟩ := Fintype.card_pos_iff.mp (l.succ_pos.trans_eq card)
  /-
    case succ
    n : Type u_4
    inst✝⁵ : DecidableEq n
    inst✝⁴ : Fintype n
    l : Nat
    ih : ∀ {R : Type u_1} {S : Type u_2} {m : Type u_5} [inst : CommRing R] [inst_ …
    R : Type u_1
    S : Type u_2
    m : Type u_5
    inst✝³ : CommRing R
    inst✝² : CommRing S
    M : Matrix m m S
    inst✝¹ : DecidableEq m
    inst✝ : Fintype m
    f : RingHom S (Matrix n n R)
    card : Eq (HAdd.hAdd l 1) (Fintype.card m)
    k : m
    ⊢ Eq (f M.det).det ((Matrix.comp m m n n R) (M.map ⇑f)).det
  -/
  let f' := f.polyToMatrix
  /-
    case succ
    n : Type u_4
    inst✝⁵ : DecidableEq n
    inst✝⁴ : Fintype n
    l : Nat
    ih : ∀ {R : Type u_1} {S : Type u_2} {m : Type u_5} [inst : CommRing R] [inst_ …
    R : Type u_1
    S : Type u_2
    m : Type u_5
    inst✝³ : CommRing R
    inst✝² : CommRing S
    M : Matrix m m S
    inst✝¹ : DecidableEq m
    inst✝ : Fintype m
    f : RingHom S (Matrix n n R)
    card : Eq (HAdd.hAdd l 1) (Fintype.card m)
    k : m
    f' : RingHom (Polynomial S) (Matrix n n (Polynomial R)) := f.polyToMatrix
    ⊢ Eq (f M.det).det ((Matrix.comp m m n n R) (M.map ⇑f)).det
  -/
  let M' := cornerAddX M k
  have : (f' M'.det).det = ((M'.map f').comp m m n n R[X]).det := by
    refine sub_eq_zero.mp <| mem_nonZeroDivisors_iff.mp
      (pow_mem ?_ _) _ (det_det_aux k fun M ↦ ih _ _ <| by simp [← card])
    rw [polyToMatrix_cornerAddX, ← charpoly]
    exact (Matrix.charpoly_monic _).mem_nonZeroDivisors
  /-
    case succ
    n : Type u_4
    inst✝⁵ : DecidableEq n
    inst✝⁴ : Fintype n
    l : Nat
    ih : ∀ {R : Type u_1} {S : Type u_2} {m : Type u_5} [inst : CommRing R] [inst_ …
    R : Type u_1
    S : Type u_2
    m : Type u_5
    inst✝³ : CommRing R
    inst✝² : CommRing S
    M : Matrix m m S
    inst✝¹ : DecidableEq m
    inst✝ : Fintype m
    f : RingHom S (Matrix n n R)
    card : Eq (HAdd.hAdd l 1) (Fintype.card m)
    k : m
    f' : RingHom (Polynomial S) (Matrix n n (Polynomial R)) := f.polyToMatrix
    M' : Matrix m m (Polynomial S) := Algebra.Norm.Transitivity.cornerAddX M k
    this : Eq (f' M'.det).det ((Matrix.comp m m n n (Polynomial R)) (M'.map ⇑f')). …
    ⊢ Eq (f M.det).det ((Matrix.comp m m n n R) (M.map ⇑f)).det
  -/
  rw [← eval_zero_det_det, congr_arg (eval 0) this, eval_zero_comp_det]
  /-
    🎉 no goals
  -/


theorem LinearMap.det_restrictScalars [AddCommGroup A] [Module R A] [Module S A]
    [IsScalarTower R S A] [Module.Free S A] {f : A →ₗ[S] A} :
    (f.restrictScalars R).det = Algebra.norm R f.det := by
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    inst✝⁵ : Module.Free R S
    inst✝⁴ : AddCommGroup A
    inst✝³ : Module R A
    inst✝² : Module S A
    inst✝¹ : IsScalarTower R S A
    inst✝ : Module.Free S A
    f : LinearMap (RingHom.id S) A A
    ⊢ Eq (LinearMap.det (↑R f)) ((Algebra.norm R) (LinearMap.det f))
  -/
  nontriviality R
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    inst✝⁵ : Module.Free R S
    inst✝⁴ : AddCommGroup A
    inst✝³ : Module R A
    inst✝² : Module S A
    inst✝¹ : IsScalarTower R S A
    inst✝ : Module.Free S A
    f : LinearMap (RingHom.id S) A A
    a✝ : Nontrivial R
    ⊢ Eq (LinearMap.det (↑R f)) ((Algebra.norm R) (LinearMap.det f))
  -/
  cases subsingleton_or_nontrivial A
    /-
      case inl
      R : Type u_1
      S : Type u_2
      A : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Algebra R S
      inst✝⁵ : Module.Free R S
      inst✝⁴ : AddCommGroup A
      inst✝³ : Module R A
      inst✝² : Module S A
      inst✝¹ : IsScalarTower R S A
      inst✝ : Module.Free S A
      f : LinearMap (RingHom.id S) A A
      a✝ : Nontrivial R
      h✝ : Subsingleton A
      ⊢ Eq (LinearMap.det (↑R f)) ((Algebra.norm R) (LinearMap.det f))
    -/
  · simp_rw [det_eq_one_of_subsingleton, _root_.map_one]
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    inst✝⁵ : Module.Free R S
    inst✝⁴ : AddCommGroup A
    inst✝³ : Module R A
    inst✝² : Module S A
    inst✝¹ : IsScalarTower R S A
    inst✝ : Module.Free S A
    f : LinearMap (RingHom.id S) A A
    a✝ : Nontrivial R
    h✝ : Nontrivial A
    ⊢ Eq (LinearMap.det (↑R f)) ((Algebra.norm R) (LinearMap.det f))
  -/
  have := Module.nontrivial S A
  /-
    case inr
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    inst✝⁵ : Module.Free R S
    inst✝⁴ : AddCommGroup A
    inst✝³ : Module R A
    inst✝² : Module S A
    inst✝¹ : IsScalarTower R S A
    inst✝ : Module.Free S A
    f : LinearMap (RingHom.id S) A A
    a✝ : Nontrivial R
    h✝ : Nontrivial A
    this : Nontrivial S
    ⊢ Eq (LinearMap.det (↑R f)) ((Algebra.norm R) (LinearMap.det f))
  -/
  let ⟨ιS, bS⟩ := Module.Free.exists_basis (R := R) (M := S)
  /-
    case inr
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    inst✝⁵ : Module.Free R S
    inst✝⁴ : AddCommGroup A
    inst✝³ : Module R A
    inst✝² : Module S A
    inst✝¹ : IsScalarTower R S A
    inst✝ : Module.Free S A
    f : LinearMap (RingHom.id S) A A
    a✝ : Nontrivial R
    h✝ : Nontrivial A
    this : Nontrivial S
    ιS : Type u_2
    bS : Basis ιS R S
    ⊢ Eq (LinearMap.det (↑R f)) ((Algebra.norm R) (LinearMap.det f))
  -/
  let ⟨ιA, bA⟩ := Module.Free.exists_basis (R := S) (M := A)
  /-
    case inr
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    inst✝⁵ : Module.Free R S
    inst✝⁴ : AddCommGroup A
    inst✝³ : Module R A
    inst✝² : Module S A
    inst✝¹ : IsScalarTower R S A
    inst✝ : Module.Free S A
    f : LinearMap (RingHom.id S) A A
    a✝ : Nontrivial R
    h✝ : Nontrivial A
    this : Nontrivial S
    ιS : Type u_2
    bS : Basis ιS R S
    ιA : Type u_3
    bA : Basis ιA S A
    ⊢ Eq (LinearMap.det (↑R f)) ((Algebra.norm R) (LinearMap.det f))
  -/
  have := bS.index_nonempty
  /-
    case inr
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    inst✝⁵ : Module.Free R S
    inst✝⁴ : AddCommGroup A
    inst✝³ : Module R A
    inst✝² : Module S A
    inst✝¹ : IsScalarTower R S A
    inst✝ : Module.Free S A
    f : LinearMap (RingHom.id S) A A
    a✝ : Nontrivial R
    h✝ : Nontrivial A
    this✝ : Nontrivial S
    ιS : Type u_2
    bS : Basis ιS R S
    ιA : Type u_3
    bA : Basis ιA S A
    this : Nonempty ιS
    ⊢ Eq (LinearMap.det (↑R f)) ((Algebra.norm R) (LinearMap.det f))
  -/
  have := bA.index_nonempty
  /-
    case inr
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    inst✝⁵ : Module.Free R S
    inst✝⁴ : AddCommGroup A
    inst✝³ : Module R A
    inst✝² : Module S A
    inst✝¹ : IsScalarTower R S A
    inst✝ : Module.Free S A
    f : LinearMap (RingHom.id S) A A
    a✝ : Nontrivial R
    h✝ : Nontrivial A
    this✝¹ : Nontrivial S
    ιS : Type u_2
    bS : Basis ιS R S
    ιA : Type u_3
    bA : Basis ιA S A
    this✝ : Nonempty ιS
    this : Nonempty ιA
    ⊢ Eq (LinearMap.det (↑R f)) ((Algebra.norm R) (LinearMap.det f))
  -/
  cases fintypeOrInfinite ιS; swap
  · rw [Algebra.norm_eq_one_of_not_module_finite (Module.not_finite_of_infinite_basis bS),
      det_eq_one_of_not_module_finite (Module.not_finite_of_infinite_basis (bS.smulTower bA))]
  /-
    case inr.inl
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    inst✝⁵ : Module.Free R S
    inst✝⁴ : AddCommGroup A
    inst✝³ : Module R A
    inst✝² : Module S A
    inst✝¹ : IsScalarTower R S A
    inst✝ : Module.Free S A
    f : LinearMap (RingHom.id S) A A
    a✝ : Nontrivial R
    h✝ : Nontrivial A
    this✝¹ : Nontrivial S
    ιS : Type u_2
    bS : Basis ιS R S
    ιA : Type u_3
    bA : Basis ιA S A
    this✝ : Nonempty ιS
    this : Nonempty ιA
    val✝ : Fintype ιS
    ⊢ Eq (LinearMap.det (↑R f)) ((Algebra.norm R) (LinearMap.det f))
  -/
  cases fintypeOrInfinite ιA; swap
  · rw [det_eq_one_of_not_module_finite (Module.not_finite_of_infinite_basis bA), _root_.map_one,
      det_eq_one_of_not_module_finite (Module.not_finite_of_infinite_basis (bS.smulTower bA))]
  classical
  rw [Algebra.norm_eq_matrix_det bS, ← AlgHom.coe_toRingHom, ← det_toMatrix bA, det_det,
    ← det_toMatrix (bS.smulTower' bA), restrictScalars_toMatrix]
  rfl


/--Let A/S/R be a tower of finite free tower of rings (with R and S commutative).
Then $\text{Norm}_{A/R} = \text{Norm}_{A/S} \circ \text{Norm}_{S/R}$.-/
theorem Algebra.norm_norm {A} [Ring A] [Algebra R A] [Algebra S A]
    [IsScalarTower R S A] [Module.Free S A] {a : A} :
    norm R (norm S a) = norm R a := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    inst✝⁵ : Module.Free R S
    A : Type u_6
    inst✝⁴ : Ring A
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    inst✝ : Module.Free S A
    a : A
    ⊢ Eq ((Algebra.norm R) ((Algebra.norm S) a)) ((Algebra.norm R) a)
  -/
  rw [norm_apply S, norm_apply R a, ← LinearMap.det_restrictScalars]; rfl
                                                                      /-
                                                                        🎉 no goals
                                                                      -/

