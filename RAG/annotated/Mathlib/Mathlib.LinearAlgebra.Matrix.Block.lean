/-- Let `b` map rows and columns of a square matrix `M` to blocks indexed by `α`s. Then
`BlockTriangular M n b` says the matrix is block triangular. -/
def BlockTriangular (M : Matrix m m R) (b : m → α) : Prop :=
  ∀ ⦃i j⦄, b j < b i → M i j = 0


@[simp]
protected theorem BlockTriangular.submatrix {f : n → m} (h : M.BlockTriangular b) :
    (M.submatrix f f).BlockTriangular (b ∘ f) := fun _ _ hij => h hij


theorem blockTriangular_reindex_iff {b : n → α} {e : m ≃ n} :
    (reindex e e M).BlockTriangular b ↔ M.BlockTriangular (b ∘ e) := by
  /-
    α : Type u_1
    m : Type u_3
    n : Type u_4
    R : Type v
    M : Matrix m m R
    inst✝¹ : LT α
    inst✝ : Zero R
    b : n → α
    e : Equiv m n
    ⊢ Iff (((Matrix.reindex e e) M).BlockTriangular b) (M.BlockTriangular (Functio …
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case refine_1
      α : Type u_1
      m : Type u_3
      n : Type u_4
      R : Type v
      M : Matrix m m R
      inst✝¹ : LT α
      inst✝ : Zero R
      b : n → α
      e : Equiv m n
      h : ((Matrix.reindex e e) M).BlockTriangular b
      ⊢ M.BlockTriangular (Function.comp b ⇑e)
    -/
  · convert h.submatrix
    /-
      case h.e'_6
      α : Type u_1
      m : Type u_3
      n : Type u_4
      R : Type v
      M : Matrix m m R
      inst✝¹ : LT α
      inst✝ : Zero R
      b : n → α
      e : Equiv m n
      h : ((Matrix.reindex e e) M).BlockTriangular b
      ⊢ Eq M (((Matrix.reindex e e) M).submatrix ⇑e ⇑e)
    -/
    simp only [reindex_apply, submatrix_submatrix, submatrix_id_id, Equiv.symm_comp_self]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m : Type u_3
      n : Type u_4
      R : Type v
      M : Matrix m m R
      inst✝¹ : LT α
      inst✝ : Zero R
      b : n → α
      e : Equiv m n
      h : M.BlockTriangular (Function.comp b ⇑e)
      ⊢ ((Matrix.reindex e e) M).BlockTriangular b
    -/
  · convert h.submatrix
    /-
      case h.e'_7
      α : Type u_1
      m : Type u_3
      n : Type u_4
      R : Type v
      M : Matrix m m R
      inst✝¹ : LT α
      inst✝ : Zero R
      b : n → α
      e : Equiv m n
      h : M.BlockTriangular (Function.comp b ⇑e)
      ⊢ Eq b (Function.comp (Function.comp b ⇑e) ⇑e.symm)
    -/
    simp only [comp_assoc b e e.symm, Equiv.self_comp_symm, comp_id]
    /-
      🎉 no goals
    -/


protected theorem BlockTriangular.transpose :
    M.BlockTriangular b → Mᵀ.BlockTriangular (toDual ∘ b) :=
  swap


@[simp]
protected theorem blockTriangular_transpose_iff {b : m → αᵒᵈ} :
    Mᵀ.BlockTriangular b ↔ M.BlockTriangular (ofDual ∘ b) :=
  forall_swap


@[simp]
theorem blockTriangular_zero : BlockTriangular (0 : Matrix m m R) b := fun _ _ _ => rfl


protected theorem BlockTriangular.neg [NegZeroClass R] {M : Matrix m m R}
    (hM : BlockTriangular M b) : BlockTriangular (-M) b :=
                  /-
                    α : Type u_1
                    m : Type u_3
                    R : Type v
                    b : m → α
                    inst✝¹ : LT α
                    inst✝ : NegZeroClass R
                    M : Matrix m m R
                    hM : M.BlockTriangular b
                    x✝¹ x✝ : m
                    h : LT.lt (b x✝) (b x✝¹)
                    ⊢ Eq (Neg.neg M x✝¹ x✝) 0
                  -/
  fun _ _ h => by rw [neg_apply, hM h, neg_zero]
                  /-
                    🎉 no goals
                  -/


theorem BlockTriangular.add [AddZeroClass R] (hM : BlockTriangular M b) (hN : BlockTriangular N b) :
                                                 /-
                                                   α : Type u_1
                                                   m : Type u_3
                                                   R : Type v
                                                   M N : Matrix m m R
                                                   b : m → α
                                                   inst✝¹ : LT α
                                                   inst✝ : AddZeroClass R
                                                   hM : M.BlockTriangular b
                                                   hN : N.BlockTriangular b
                                                   i j : m
                                                   h : LT.lt (b j) (b i)
                                                   ⊢ Eq (HAdd.hAdd M N i j) 0
                                                 -/
    BlockTriangular (M + N) b := fun i j h => by simp_rw [Matrix.add_apply, hM h, hN h, zero_add]
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem BlockTriangular.sub [SubNegZeroMonoid R]
    (hM : BlockTriangular M b) (hN : BlockTriangular N b) :
                                                 /-
                                                   α : Type u_1
                                                   m : Type u_3
                                                   R : Type v
                                                   M N : Matrix m m R
                                                   b : m → α
                                                   inst✝¹ : LT α
                                                   inst✝ : SubNegZeroMonoid R
                                                   hM : M.BlockTriangular b
                                                   hN : N.BlockTriangular b
                                                   i j : m
                                                   h : LT.lt (b j) (b i)
                                                   ⊢ Eq (HSub.hSub M N i j) 0
                                                 -/
    BlockTriangular (M - N) b := fun i j h => by simp_rw [Matrix.sub_apply, hM h, hN h, sub_zero]
                                                 /-
                                                   🎉 no goals
                                                 -/


lemma BlockTriangular.add_iff_right [AddGroup R] (hM : BlockTriangular M b) :
    BlockTriangular (M + N) b ↔ BlockTriangular N b := ⟨(by simpa using hM.neg.add ·), hM.add⟩


lemma BlockTriangular.add_iff_left [AddGroup R] (hN : BlockTriangular N b) :
    BlockTriangular (M + N) b ↔ BlockTriangular M b := ⟨(by simpa using ·.sub hN), (·.add hN)⟩


lemma BlockTriangular.sub_iff_right [AddGroup R] (hM : BlockTriangular M b) :
    BlockTriangular (M - N) b ↔ BlockTriangular N b := ⟨(by simpa using ·.neg.add hM), hM.sub⟩


lemma BlockTriangular.sub_iff_left [AddGroup R] (hN : BlockTriangular N b) :
    BlockTriangular (M - N) b ↔ BlockTriangular M b := ⟨(by simpa using ·.add hN), (·.sub hN)⟩


lemma BlockTriangular.map {S F} [FunLike F R S] [Zero R] [Zero S] [ZeroHomClass F R S] (f : F)
    (h : BlockTriangular M b) : BlockTriangular (M.map f) b :=
                  /-
                    α : Type u_1
                    m : Type u_3
                    R : Type v
                    M : Matrix m m R
                    b : m → α
                    inst✝⁴ : LT α
                    S : Type u_8
                    F : Type u_9
                    inst✝³ : FunLike F R S
                    inst✝² : Zero R
                    inst✝¹ : Zero S
                    inst✝ : ZeroHomClass F R S
                    f : F
                    h : M.BlockTriangular b
                    i j : m
                    lt : LT.lt (b j) (b i)
                    ⊢ Eq (M.map (⇑f) i j) 0
                  -/
  fun i j lt ↦ by simp [h lt]
                  /-
                    🎉 no goals
                  -/


lemma BlockTriangular.comp [Zero R] {M : Matrix m m (Matrix n n R)} (h : BlockTriangular M b) :
    BlockTriangular (M.comp m m n n R) fun i ↦ b i.1 :=
                  /-
                    α : Type u_1
                    m : Type u_3
                    n : Type u_4
                    R : Type v
                    b : m → α
                    inst✝¹ : LT α
                    inst✝ : Zero R
                    M : Matrix m m (Matrix n n R)
                    h : M.BlockTriangular b
                    i j : Prod m n
                    lt : LT.lt ((fun i => b i.1) j) ((fun i => b i.1) i)
                    ⊢ Eq ((Matrix.comp m m n n R) M i j) 0
                  -/
  fun i j lt ↦ by simp [h lt]
                  /-
                    🎉 no goals
                  -/


theorem blockTriangular_diagonal [DecidableEq m] (d : m → R) : BlockTriangular (diagonal d) b :=
  fun _ _ h => diagonal_apply_ne' d fun h' => ne_of_lt h (congr_arg _ h')


theorem blockTriangular_blockDiagonal' [DecidableEq α] (d : ∀ i : α, Matrix (m' i) (m' i) R) :
    BlockTriangular (blockDiagonal' d) Sigma.fst := by
  /-
    α : Type u_1
    m' : α → Type u_6
    R : Type v
    inst✝² : Preorder α
    inst✝¹ : Zero R
    inst✝ : DecidableEq α
    d : (i : α) → Matrix (m' i) (m' i) R
    ⊢ (Matrix.blockDiagonal' d).BlockTriangular Sigma.fst
  -/
  rintro ⟨i, i'⟩ ⟨j, j'⟩ h
  /-
    case mk.mk
    α : Type u_1
    m' : α → Type u_6
    R : Type v
    inst✝² : Preorder α
    inst✝¹ : Zero R
    inst✝ : DecidableEq α
    d : (i : α) → Matrix (m' i) (m' i) R
    i : α
    i' : m' i
    j : α
    j' : m' j
    h : LT.lt ⟨j, j'⟩.fst ⟨i, i'⟩.fst
    ⊢ Eq (Matrix.blockDiagonal' d ⟨i, i'⟩ ⟨j, j'⟩) 0
  -/
  apply blockDiagonal'_apply_ne d i' j' fun h' => ne_of_lt h h'.symm
  /-
    🎉 no goals
  -/


theorem blockTriangular_blockDiagonal [DecidableEq α] (d : α → Matrix m m R) :
    BlockTriangular (blockDiagonal d) Prod.snd := by
  /-
    α : Type u_1
    m : Type u_3
    R : Type v
    inst✝² : Preorder α
    inst✝¹ : Zero R
    inst✝ : DecidableEq α
    d : α → Matrix m m R
    ⊢ (Matrix.blockDiagonal d).BlockTriangular Prod.snd
  -/
  rintro ⟨i, i'⟩ ⟨j, j'⟩ h
  /-
    case mk.mk
    α : Type u_1
    m : Type u_3
    R : Type v
    inst✝² : Preorder α
    inst✝¹ : Zero R
    inst✝ : DecidableEq α
    d : α → Matrix m m R
    i : m
    i' : α
    j : m
    j' : α
    h : LT.lt { fst := j, snd := j' }.2 { fst := i, snd := i' }.2
    ⊢ Eq (Matrix.blockDiagonal d { fst := i, snd := i' } { fst := j, snd := j' }) 0
  -/
  rw [blockDiagonal'_eq_blockDiagonal, blockTriangular_blockDiagonal']
  /-
    case mk.mk.a
    α : Type u_1
    m : Type u_3
    R : Type v
    inst✝² : Preorder α
    inst✝¹ : Zero R
    inst✝ : DecidableEq α
    d : α → Matrix m m R
    i : m
    i' : α
    j : m
    j' : α
    h : LT.lt { fst := j, snd := j' }.2 { fst := i, snd := i' }.2
    ⊢ LT.lt ⟨j', j⟩.fst ⟨i', i⟩.fst
  -/
  exact h
  /-
    🎉 no goals
  -/


theorem blockTriangular_one [One R] : BlockTriangular (1 : Matrix m m R) b :=
  blockTriangular_diagonal _


theorem blockTriangular_stdBasisMatrix {i j : m} (hij : b i ≤ b j) (c : R) :
    BlockTriangular (stdBasisMatrix i j c) b := by
  /-
    α : Type u_1
    m : Type u_3
    R : Type v
    b : m → α
    inst✝² : Preorder α
    inst✝¹ : Zero R
    inst✝ : DecidableEq m
    i j : m
    hij : LE.le (b i) (b j)
    c : R
    ⊢ (Matrix.stdBasisMatrix i j c).BlockTriangular b
  -/
  intro r s hrs
  /-
    α : Type u_1
    m : Type u_3
    R : Type v
    b : m → α
    inst✝² : Preorder α
    inst✝¹ : Zero R
    inst✝ : DecidableEq m
    i j : m
    hij : LE.le (b i) (b j)
    c : R
    r s : m
    hrs : LT.lt (b s) (b r)
    ⊢ Eq (Matrix.stdBasisMatrix i j c r s) 0
  -/
  apply StdBasisMatrix.apply_of_ne
  /-
    case h
    α : Type u_1
    m : Type u_3
    R : Type v
    b : m → α
    inst✝² : Preorder α
    inst✝¹ : Zero R
    inst✝ : DecidableEq m
    i j : m
    hij : LE.le (b i) (b j)
    c : R
    r s : m
    hrs : LT.lt (b s) (b r)
    ⊢ Not (And (Eq i r) (Eq j s))
  -/
  rintro ⟨rfl, rfl⟩
  /-
    case h.intro
    α : Type u_1
    m : Type u_3
    R : Type v
    b : m → α
    inst✝² : Preorder α
    inst✝¹ : Zero R
    inst✝ : DecidableEq m
    i j : m
    hij : LE.le (b i) (b j)
    c : R
    hrs : LT.lt (b j) (b i)
    ⊢ False
  -/
  exact (hij.trans_lt hrs).false
  /-
    🎉 no goals
  -/


theorem blockTriangular_stdBasisMatrix' {i j : m} (hij : b j ≤ b i) (c : R) :
    BlockTriangular (stdBasisMatrix i j c) (toDual ∘ b) :=
                                     /-
                                       α : Type u_1
                                       m : Type u_3
                                       R : Type v
                                       b : m → α
                                       inst✝² : Preorder α
                                       inst✝¹ : Zero R
                                       inst✝ : DecidableEq m
                                       i j : m
                                       hij : LE.le (b j) (b i)
                                       c : R
                                       ⊢ LE.le (Function.comp (⇑OrderDual.toDual) b i) (Function.comp (⇑OrderDual.toD …
                                     -/
  blockTriangular_stdBasisMatrix (by exact toDual_le_toDual.mpr hij) _
                                     /-
                                       🎉 no goals
                                     -/


theorem blockTriangular_transvection {i j : m} (hij : b i ≤ b j) (c : R) :
    BlockTriangular (transvection i j c) b :=
  blockTriangular_one.add (blockTriangular_stdBasisMatrix hij c)


theorem blockTriangular_transvection' {i j : m} (hij : b j ≤ b i) (c : R) :
    BlockTriangular (transvection i j c) (OrderDual.toDual ∘ b) :=
  blockTriangular_one.add (blockTriangular_stdBasisMatrix' hij c)


theorem BlockTriangular.mul [Fintype m] [NonUnitalNonAssocSemiring R]
    {M N : Matrix m m R} (hM : BlockTriangular M b)
    (hN : BlockTriangular N b) : BlockTriangular (M * N) b := by
  /-
    α : Type u_1
    m : Type u_3
    R : Type v
    b : m → α
    inst✝² : LinearOrder α
    inst✝¹ : Fintype m
    inst✝ : NonUnitalNonAssocSemiring R
    M N : Matrix m m R
    hM : M.BlockTriangular b
    hN : N.BlockTriangular b
    ⊢ (HMul.hMul M N).BlockTriangular b
  -/
  intro i j hij
  /-
    α : Type u_1
    m : Type u_3
    R : Type v
    b : m → α
    inst✝² : LinearOrder α
    inst✝¹ : Fintype m
    inst✝ : NonUnitalNonAssocSemiring R
    M N : Matrix m m R
    hM : M.BlockTriangular b
    hN : N.BlockTriangular b
    i j : m
    hij : LT.lt (b j) (b i)
    ⊢ Eq (HMul.hMul M N i j) 0
  -/
  apply Finset.sum_eq_zero
  /-
    case h
    α : Type u_1
    m : Type u_3
    R : Type v
    b : m → α
    inst✝² : LinearOrder α
    inst✝¹ : Fintype m
    inst✝ : NonUnitalNonAssocSemiring R
    M N : Matrix m m R
    hM : M.BlockTriangular b
    hN : N.BlockTriangular b
    i j : m
    hij : LT.lt (b j) (b i)
    ⊢ ∀ (x : m), Membership.mem Finset.univ x → Eq (HMul.hMul ((fun j => M i j) x) …
  -/
  intro k _
  /-
    case h
    α : Type u_1
    m : Type u_3
    R : Type v
    b : m → α
    inst✝² : LinearOrder α
    inst✝¹ : Fintype m
    inst✝ : NonUnitalNonAssocSemiring R
    M N : Matrix m m R
    hM : M.BlockTriangular b
    hN : N.BlockTriangular b
    i j : m
    hij : LT.lt (b j) (b i)
    k : m
    a✝ : Membership.mem Finset.univ k
    ⊢ Eq (HMul.hMul ((fun j => M i j) k) ((fun j_1 => N j_1 j) k)) 0
  -/
  by_cases hki : b k < b i
    /-
      case pos
      α : Type u_1
      m : Type u_3
      R : Type v
      b : m → α
      inst✝² : LinearOrder α
      inst✝¹ : Fintype m
      inst✝ : NonUnitalNonAssocSemiring R
      M N : Matrix m m R
      hM : M.BlockTriangular b
      hN : N.BlockTriangular b
      i j : m
      hij : LT.lt (b j) (b i)
      k : m
      a✝ : Membership.mem Finset.univ k
      hki : LT.lt (b k) (b i)
      ⊢ Eq (HMul.hMul ((fun j => M i j) k) ((fun j_1 => N j_1 j) k)) 0
    -/
  · simp_rw [hM hki, zero_mul]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : Type u_3
      R : Type v
      b : m → α
      inst✝² : LinearOrder α
      inst✝¹ : Fintype m
      inst✝ : NonUnitalNonAssocSemiring R
      M N : Matrix m m R
      hM : M.BlockTriangular b
      hN : N.BlockTriangular b
      i j : m
      hij : LT.lt (b j) (b i)
      k : m
      a✝ : Membership.mem Finset.univ k
      hki : Not (LT.lt (b k) (b i))
      ⊢ Eq (HMul.hMul ((fun j => M i j) k) ((fun j_1 => N j_1 j) k)) 0
    -/
  · simp_rw [hN (lt_of_lt_of_le hij (le_of_not_lt hki)), mul_zero]
    /-
      🎉 no goals
    -/


theorem upper_two_blockTriangular [Zero R] [Preorder α] (A : Matrix m m R) (B : Matrix m n R)
    (D : Matrix n n R) {a b : α} (hab : a < b) :
    BlockTriangular (fromBlocks A B 0 D) (Sum.elim (fun _ => a) fun _ => b) := by
  /-
    α : Type u_1
    m : Type u_3
    n : Type u_4
    R : Type v
    inst✝¹ : Zero R
    inst✝ : Preorder α
    A : Matrix m m R
    B : Matrix m n R
    D : Matrix n n R
    a b : α
    hab : LT.lt a b
    ⊢ (Matrix.fromBlocks A B 0 D).BlockTriangular (Sum.elim (fun x => a) fun x => b)
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
  rintro (c | c) (d | d) hcd <;> first | simp [hab.not_lt] at hcd ⊢
                                 /-
                                   🎉 no goals
                                 -/


theorem equiv_block_det (M : Matrix m m R) {p q : m → Prop} [DecidablePred p] [DecidablePred q]
    (e : ∀ x, q x ↔ p x) : (toSquareBlockProp M p).det = (toSquareBlockProp M q).det := by
  /-
    m : Type u_3
    R : Type v
    inst✝⁴ : CommRing R
    inst✝³ : DecidableEq m
    inst✝² : Fintype m
    M : Matrix m m R
    p q : m → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidablePred q
    e : ∀ (x : m), Iff (q x) (p x)
    ⊢ Eq (M.toSquareBlockProp p).det (M.toSquareBlockProp q).det
  -/
  convert Matrix.det_reindex_self (Equiv.subtypeEquivRight e) (toSquareBlockProp M q)
  /-
    🎉 no goals
  -/

-- Removed `@[simp]` attribute,
-- as the LHS simplifies already to `M.toSquareBlock id i ⟨i, ⋯⟩ ⟨i, ⋯⟩`

theorem det_toSquareBlock_id (M : Matrix m m R) (i : m) : (M.toSquareBlock id i).det = M i i :=
  letI : Unique { a // id a = i } := ⟨⟨⟨i, rfl⟩⟩, fun j => Subtype.ext j.property⟩
  (det_unique _).trans rfl


theorem det_toBlock (M : Matrix m m R) (p : m → Prop) [DecidablePred p] :
    M.det =
      (fromBlocks (toBlock M p p) (toBlock M p fun j => ¬p j) (toBlock M (fun j => ¬p j) p) <|
          toBlock M (fun j => ¬p j) fun j => ¬p j).det := by
  /-
    m : Type u_3
    R : Type v
    inst✝³ : CommRing R
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    M : Matrix m m R
    p : m → Prop
    inst✝ : DecidablePred p
    ⊢ Eq M.det (Matrix.fromBlocks (M.toBlock p p) (M.toBlock p fun j => Not (p j)) …
  -/
  rw [← Matrix.det_reindex_self (Equiv.sumCompl p).symm M]
  /-
    m : Type u_3
    R : Type v
    inst✝³ : CommRing R
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    M : Matrix m m R
    p : m → Prop
    inst✝ : DecidablePred p
    ⊢ Eq ((Matrix.reindex (Equiv.sumCompl p).symm (Equiv.sumCompl p).symm) M).det  …
  -/
  rw [det_apply', det_apply']
  /-
    m : Type u_3
    R : Type v
    inst✝³ : CommRing R
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    M : Matrix m m R
    p : m → Prop
    inst✝ : DecidablePred p
    ⊢ Eq (Finset.univ.sum fun σ => HMul.hMul (↑↑(Equiv.Perm.sign σ)) (Finset.univ. …
  -/
  congr; ext σ; congr; ext x
  /-
    case e_f.h.e_a.e_f.h
    m : Type u_3
    R : Type v
    inst✝³ : CommRing R
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    M : Matrix m m R
    p : m → Prop
    inst✝ : DecidablePred p
    σ : Equiv.Perm (Sum (Subtype fun a => p a) (Subtype fun a => Not (p a)))
    x : Sum (Subtype fun a => p a) (Subtype fun a => Not (p a))
    ⊢ Eq ((Matrix.reindex (Equiv.sumCompl p).symm (Equiv.sumCompl p).symm) M (σ x) …
  -/
  generalize hy : σ x = y
  /-
    case e_f.h.e_a.e_f.h
    m : Type u_3
    R : Type v
    inst✝³ : CommRing R
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    M : Matrix m m R
    p : m → Prop
    inst✝ : DecidablePred p
    σ : Equiv.Perm (Sum (Subtype fun a => p a) (Subtype fun a => Not (p a)))
    x y : Sum (Subtype fun a => p a) (Subtype fun a => Not (p a))
    hy : Eq (σ x) y
    ⊢ Eq ((Matrix.reindex (Equiv.sumCompl p).symm (Equiv.sumCompl p).symm) M y x)  …
  -/
  cases x <;> cases y <;>
    simp only [Matrix.reindex_apply, toBlock_apply, Equiv.symm_symm, Equiv.sumCompl_apply_inr,
      Equiv.sumCompl_apply_inl, fromBlocks_apply₁₁, fromBlocks_apply₁₂, fromBlocks_apply₂₁,
      fromBlocks_apply₂₂, Matrix.submatrix_apply]


theorem twoBlockTriangular_det (M : Matrix m m R) (p : m → Prop) [DecidablePred p]
    (h : ∀ i, ¬p i → ∀ j, p j → M i j = 0) :
    M.det = (toSquareBlockProp M p).det * (toSquareBlockProp M fun i => ¬p i).det := by
  /-
    m : Type u_3
    R : Type v
    inst✝³ : CommRing R
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    M : Matrix m m R
    p : m → Prop
    inst✝ : DecidablePred p
    h : ∀ (i : m), Not (p i) → ∀ (j : m), p j → Eq (M i j) 0
    ⊢ Eq M.det (HMul.hMul (M.toSquareBlockProp p).det (M.toSquareBlockProp fun i = …
  -/
  rw [det_toBlock M p]
  convert det_fromBlocks_zero₂₁ (toBlock M p p) (toBlock M p fun j => ¬p j)
      (toBlock M (fun j => ¬p j) fun j => ¬p j)
  /-
    case h.e'_2.h.e'_6.h.e'_8
    m : Type u_3
    R : Type v
    inst✝³ : CommRing R
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    M : Matrix m m R
    p : m → Prop
    inst✝ : DecidablePred p
    h : ∀ (i : m), Not (p i) → ∀ (j : m), p j → Eq (M i j) 0
    ⊢ Eq (M.toBlock (fun j => Not (p j)) p) 0
  -/
  ext i j
  /-
    case h.e'_2.h.e'_6.h.e'_8.a
    m : Type u_3
    R : Type v
    inst✝³ : CommRing R
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    M : Matrix m m R
    p : m → Prop
    inst✝ : DecidablePred p
    h : ∀ (i : m), Not (p i) → ∀ (j : m), p j → Eq (M i j) 0
    i : Subtype fun a => Not (p a)
    j : Subtype fun a => p a
    ⊢ Eq (M.toBlock (fun j => Not (p j)) p i j) (0 i j)
  -/
  exact h (↑i) i.2 (↑j) j.2
  /-
    🎉 no goals
  -/


theorem twoBlockTriangular_det' (M : Matrix m m R) (p : m → Prop) [DecidablePred p]
    (h : ∀ i, p i → ∀ j, ¬p j → M i j = 0) :
    M.det = (toSquareBlockProp M p).det * (toSquareBlockProp M fun i => ¬p i).det := by
  /-
    m : Type u_3
    R : Type v
    inst✝³ : CommRing R
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    M : Matrix m m R
    p : m → Prop
    inst✝ : DecidablePred p
    h : ∀ (i : m), p i → ∀ (j : m), Not (p j) → Eq (M i j) 0
    ⊢ Eq M.det (HMul.hMul (M.toSquareBlockProp p).det (M.toSquareBlockProp fun i = …
  -/
  rw [M.twoBlockTriangular_det fun i => ¬p i, mul_comm]
    /-
      m : Type u_3
      R : Type v
      inst✝³ : CommRing R
      inst✝² : DecidableEq m
      inst✝¹ : Fintype m
      M : Matrix m m R
      p : m → Prop
      inst✝ : DecidablePred p
      h : ∀ (i : m), p i → ∀ (j : m), Not (p j) → Eq (M i j) 0
      ⊢ Eq (HMul.hMul (M.toSquareBlockProp fun i => Not (Not (p i))).det (M.toSquare …
    -/
  · congr 1
    /-
      case e_a
      m : Type u_3
      R : Type v
      inst✝³ : CommRing R
      inst✝² : DecidableEq m
      inst✝¹ : Fintype m
      M : Matrix m m R
      p : m → Prop
      inst✝ : DecidablePred p
      h : ∀ (i : m), p i → ∀ (j : m), Not (p j) → Eq (M i j) 0
      ⊢ Eq (M.toSquareBlockProp fun i => Not (Not (p i))).det (M.toSquareBlockProp p …
    -/
    exact equiv_block_det _ fun _ => not_not.symm
    /-
      🎉 no goals
    -/
    /-
      m : Type u_3
      R : Type v
      inst✝³ : CommRing R
      inst✝² : DecidableEq m
      inst✝¹ : Fintype m
      M : Matrix m m R
      p : m → Prop
      inst✝ : DecidablePred p
      h : ∀ (i : m), p i → ∀ (j : m), Not (p j) → Eq (M i j) 0
      ⊢ ∀ (i : m), Not (Not (p i)) → ∀ (j : m), Not (p j) → Eq (M i j) 0
    -/
  · simpa only [Classical.not_not] using h
    /-
      🎉 no goals
    -/


protected theorem BlockTriangular.det [DecidableEq α] [LinearOrder α] (hM : BlockTriangular M b) :
    M.det = ∏ a ∈ univ.image b, (M.toSquareBlock b a).det := by
  /-
    α : Type u_1
    m : Type u_3
    R : Type v
    M : Matrix m m R
    b : m → α
    inst✝⁴ : CommRing R
    inst✝³ : DecidableEq m
    inst✝² : Fintype m
    inst✝¹ : DecidableEq α
    inst✝ : LinearOrder α
    hM : M.BlockTriangular b
    ⊢ Eq M.det ((Finset.image b Finset.univ).prod fun a => (M.toSquareBlock b a).d …
  -/
  induction' hs : univ.image b using Finset.strongInduction with s ih generalizing m
  /-
    case H
    α : Type u_1
    R : Type v
    inst✝⁴ : CommRing R
    inst✝³ : DecidableEq α
    inst✝² : LinearOrder α
    s : Finset α
    ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {m : Type u_3} {M : Matrix m …
    m : Type u_3
    M : Matrix m m R
    b : m → α
    inst✝¹ : DecidableEq m
    inst✝ : Fintype m
    hM : M.BlockTriangular b
    hs : Eq (Finset.image b Finset.univ) s
    ⊢ Eq M.det (s.prod fun a => (M.toSquareBlock b a).det)
  -/
  subst hs
  /-
    case H
    α : Type u_1
    R : Type v
    inst✝⁴ : CommRing R
    inst✝³ : DecidableEq α
    inst✝² : LinearOrder α
    m : Type u_3
    M : Matrix m m R
    b : m → α
    inst✝¹ : DecidableEq m
    inst✝ : Fintype m
    hM : M.BlockTriangular b
    ih : ∀ (t : Finset α), HasSSubset.SSubset t (Finset.image b Finset.univ) → ∀ { …
    ⊢ Eq M.det ((Finset.image b Finset.univ).prod fun a => (M.toSquareBlock b a).d …
  -/
  cases isEmpty_or_nonempty m
    /-
      case H.inl
      α : Type u_1
      R : Type v
      inst✝⁴ : CommRing R
      inst✝³ : DecidableEq α
      inst✝² : LinearOrder α
      m : Type u_3
      M : Matrix m m R
      b : m → α
      inst✝¹ : DecidableEq m
      inst✝ : Fintype m
      hM : M.BlockTriangular b
      ih : ∀ (t : Finset α), HasSSubset.SSubset t (Finset.image b Finset.univ) → ∀ { …
      h✝ : IsEmpty m
      ⊢ Eq M.det ((Finset.image b Finset.univ).prod fun a => (M.toSquareBlock b a).d …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case H.inr
    α : Type u_1
    R : Type v
    inst✝⁴ : CommRing R
    inst✝³ : DecidableEq α
    inst✝² : LinearOrder α
    m : Type u_3
    M : Matrix m m R
    b : m → α
    inst✝¹ : DecidableEq m
    inst✝ : Fintype m
    hM : M.BlockTriangular b
    ih : ∀ (t : Finset α), HasSSubset.SSubset t (Finset.image b Finset.univ) → ∀ { …
    h✝ : Nonempty m
    ⊢ Eq M.det ((Finset.image b Finset.univ).prod fun a => (M.toSquareBlock b a).d …
  -/
  let k := (univ.image b).max' (univ_nonempty.image _)
  /-
    case H.inr
    α : Type u_1
    R : Type v
    inst✝⁴ : CommRing R
    inst✝³ : DecidableEq α
    inst✝² : LinearOrder α
    m : Type u_3
    M : Matrix m m R
    b : m → α
    inst✝¹ : DecidableEq m
    inst✝ : Fintype m
    hM : M.BlockTriangular b
    ih : ∀ (t : Finset α), HasSSubset.SSubset t (Finset.image b Finset.univ) → ∀ { …
    h✝ : Nonempty m
    k : α := (Finset.image b Finset.univ).max' ⋯
    ⊢ Eq M.det ((Finset.image b Finset.univ).prod fun a => (M.toSquareBlock b a).d …
  -/
  rw [twoBlockTriangular_det' M fun i => b i = k]
  · have : univ.image b = insert k ((univ.image b).erase k) := by
      rw [insert_erase]
      apply max'_mem
    /-
      case H.inr
      α : Type u_1
      R : Type v
      inst✝⁴ : CommRing R
      inst✝³ : DecidableEq α
      inst✝² : LinearOrder α
      m : Type u_3
      M : Matrix m m R
      b : m → α
      inst✝¹ : DecidableEq m
      inst✝ : Fintype m
      hM : M.BlockTriangular b
      ih : ∀ (t : Finset α), HasSSubset.SSubset t (Finset.image b Finset.univ) → ∀ { …
      h✝ : Nonempty m
      k : α := (Finset.image b Finset.univ).max' ⋯
      this : Eq (Finset.image b Finset.univ) (Insert.insert k ((Finset.image b Finse …
      ⊢ Eq (HMul.hMul (M.toSquareBlockProp fun i => Eq (b i) k).det (M.toSquareBlock …
    -/
    rw [this, prod_insert (not_mem_erase _ _)]
    /-
      case H.inr
      α : Type u_1
      R : Type v
      inst✝⁴ : CommRing R
      inst✝³ : DecidableEq α
      inst✝² : LinearOrder α
      m : Type u_3
      M : Matrix m m R
      b : m → α
      inst✝¹ : DecidableEq m
      inst✝ : Fintype m
      hM : M.BlockTriangular b
      ih : ∀ (t : Finset α), HasSSubset.SSubset t (Finset.image b Finset.univ) → ∀ { …
      h✝ : Nonempty m
      k : α := (Finset.image b Finset.univ).max' ⋯
      this : Eq (Finset.image b Finset.univ) (Insert.insert k ((Finset.image b Finse …
      ⊢ Eq (HMul.hMul (M.toSquareBlockProp fun i => Eq (b i) k).det (M.toSquareBlock …
    -/
    refine congr_arg _ ?_
    /-
      case H.inr
      α : Type u_1
      R : Type v
      inst✝⁴ : CommRing R
      inst✝³ : DecidableEq α
      inst✝² : LinearOrder α
      m : Type u_3
      M : Matrix m m R
      b : m → α
      inst✝¹ : DecidableEq m
      inst✝ : Fintype m
      hM : M.BlockTriangular b
      ih : ∀ (t : Finset α), HasSSubset.SSubset t (Finset.image b Finset.univ) → ∀ { …
      h✝ : Nonempty m
      k : α := (Finset.image b Finset.univ).max' ⋯
      this : Eq (Finset.image b Finset.univ) (Insert.insert k ((Finset.image b Finse …
      ⊢ Eq (M.toSquareBlockProp fun i => Not (Eq (b i) k)).det (((Finset.image b Fin …
    -/
    let b' := fun i : { a // b a ≠ k } => b ↑i
    /-
      case H.inr
      α : Type u_1
      R : Type v
      inst✝⁴ : CommRing R
      inst✝³ : DecidableEq α
      inst✝² : LinearOrder α
      m : Type u_3
      M : Matrix m m R
      b : m → α
      inst✝¹ : DecidableEq m
      inst✝ : Fintype m
      hM : M.BlockTriangular b
      ih : ∀ (t : Finset α), HasSSubset.SSubset t (Finset.image b Finset.univ) → ∀ { …
      h✝ : Nonempty m
      k : α := (Finset.image b Finset.univ).max' ⋯
      this : Eq (Finset.image b Finset.univ) (Insert.insert k ((Finset.image b Finse …
      b' : (Subtype fun a => Ne (b a) k) → α := fun i => b ↑i
      ⊢ Eq (M.toSquareBlockProp fun i => Not (Eq (b i) k)).det (((Finset.image b Fin …
    -/
    have h' : BlockTriangular (M.toSquareBlockProp fun i => b i ≠ k) b' := hM.submatrix
    have hb' : image b' univ = (image b univ).erase k := by
      convert image_subtype_ne_univ_eq_image_erase k b
    /-
      case H.inr
      α : Type u_1
      R : Type v
      inst✝⁴ : CommRing R
      inst✝³ : DecidableEq α
      inst✝² : LinearOrder α
      m : Type u_3
      M : Matrix m m R
      b : m → α
      inst✝¹ : DecidableEq m
      inst✝ : Fintype m
      hM : M.BlockTriangular b
      ih : ∀ (t : Finset α), HasSSubset.SSubset t (Finset.image b Finset.univ) → ∀ { …
      h✝ : Nonempty m
      k : α := (Finset.image b Finset.univ).max' ⋯
      this : Eq (Finset.image b Finset.univ) (Insert.insert k ((Finset.image b Finse …
      b' : (Subtype fun a => Ne (b a) k) → α := fun i => b ↑i
      h' : (M.toSquareBlockProp fun i => Ne (b i) k).BlockTriangular b'
      hb' : Eq (Finset.image b' Finset.univ) ((Finset.image b Finset.univ).erase k)
      ⊢ Eq (M.toSquareBlockProp fun i => Not (Eq (b i) k)).det (((Finset.image b Fin …
    -/
    rw [ih _ (erase_ssubset <| max'_mem _ _) h' hb']
    /-
      case H.inr
      α : Type u_1
      R : Type v
      inst✝⁴ : CommRing R
      inst✝³ : DecidableEq α
      inst✝² : LinearOrder α
      m : Type u_3
      M : Matrix m m R
      b : m → α
      inst✝¹ : DecidableEq m
      inst✝ : Fintype m
      hM : M.BlockTriangular b
      ih : ∀ (t : Finset α), HasSSubset.SSubset t (Finset.image b Finset.univ) → ∀ { …
      h✝ : Nonempty m
      k : α := (Finset.image b Finset.univ).max' ⋯
      this : Eq (Finset.image b Finset.univ) (Insert.insert k ((Finset.image b Finse …
      b' : (Subtype fun a => Ne (b a) k) → α := fun i => b ↑i
      h' : (M.toSquareBlockProp fun i => Ne (b i) k).BlockTriangular b'
      hb' : Eq (Finset.image b' Finset.univ) ((Finset.image b Finset.univ).erase k)
      ⊢ Eq (((Finset.image b Finset.univ).erase ((Finset.image b Finset.univ).max' ⋯ …
    -/
    refine Finset.prod_congr rfl fun l hl => ?_
    let he : { a // b' a = l } ≃ { a // b a = l } :=
      haveI hc : ∀ i, b i = l → b i ≠ k := fun i hi => ne_of_eq_of_ne hi (ne_of_mem_erase hl)
      Equiv.subtypeSubtypeEquivSubtype @(hc)
    /-
      case H.inr
      α : Type u_1
      R : Type v
      inst✝⁴ : CommRing R
      inst✝³ : DecidableEq α
      inst✝² : LinearOrder α
      m : Type u_3
      M : Matrix m m R
      b : m → α
      inst✝¹ : DecidableEq m
      inst✝ : Fintype m
      hM : M.BlockTriangular b
      ih : ∀ (t : Finset α), HasSSubset.SSubset t (Finset.image b Finset.univ) → ∀ { …
      h✝ : Nonempty m
      k : α := (Finset.image b Finset.univ).max' ⋯
      this : Eq (Finset.image b Finset.univ) (Insert.insert k ((Finset.image b Finse …
      b' : (Subtype fun a => Ne (b a) k) → α := fun i => b ↑i
      h' : (M.toSquareBlockProp fun i => Ne (b i) k).BlockTriangular b'
      hb' : Eq (Finset.image b' Finset.univ) ((Finset.image b Finset.univ).erase k)
      l : α
      hl : Membership.mem ((Finset.image b Finset.univ).erase k) l
      he : Equiv (Subtype fun a => Eq (b' a) l) (Subtype fun a => Eq (b a) l) := Equ …
      ⊢ Eq ((M.toSquareBlockProp fun i => Ne (b i) k).toSquareBlock b' l).det (M.toS …
    -/
    simp only [toSquareBlock_def]
    /-
      case H.inr
      α : Type u_1
      R : Type v
      inst✝⁴ : CommRing R
      inst✝³ : DecidableEq α
      inst✝² : LinearOrder α
      m : Type u_3
      M : Matrix m m R
      b : m → α
      inst✝¹ : DecidableEq m
      inst✝ : Fintype m
      hM : M.BlockTriangular b
      ih : ∀ (t : Finset α), HasSSubset.SSubset t (Finset.image b Finset.univ) → ∀ { …
      h✝ : Nonempty m
      k : α := (Finset.image b Finset.univ).max' ⋯
      this : Eq (Finset.image b Finset.univ) (Insert.insert k ((Finset.image b Finse …
      b' : (Subtype fun a => Ne (b a) k) → α := fun i => b ↑i
      h' : (M.toSquareBlockProp fun i => Ne (b i) k).BlockTriangular b'
      hb' : Eq (Finset.image b' Finset.univ) ((Finset.image b Finset.univ).erase k)
      l : α
      hl : Membership.mem ((Finset.image b Finset.univ).erase k) l
      he : Equiv (Subtype fun a => Eq (b' a) l) (Subtype fun a => Eq (b a) l) := Equ …
      ⊢ Eq (Matrix.of fun i j => M.toSquareBlockProp (fun i => Ne (b i) k) ↑i ↑j).de …
    -/
    erw [← Matrix.det_reindex_self he.symm fun i j : { a // b a = l } => M ↑i ↑j]
    /-
      case H.inr
      α : Type u_1
      R : Type v
      inst✝⁴ : CommRing R
      inst✝³ : DecidableEq α
      inst✝² : LinearOrder α
      m : Type u_3
      M : Matrix m m R
      b : m → α
      inst✝¹ : DecidableEq m
      inst✝ : Fintype m
      hM : M.BlockTriangular b
      ih : ∀ (t : Finset α), HasSSubset.SSubset t (Finset.image b Finset.univ) → ∀ { …
      h✝ : Nonempty m
      k : α := (Finset.image b Finset.univ).max' ⋯
      this : Eq (Finset.image b Finset.univ) (Insert.insert k ((Finset.image b Finse …
      b' : (Subtype fun a => Ne (b a) k) → α := fun i => b ↑i
      h' : (M.toSquareBlockProp fun i => Ne (b i) k).BlockTriangular b'
      hb' : Eq (Finset.image b' Finset.univ) ((Finset.image b Finset.univ).erase k)
      l : α
      hl : Membership.mem ((Finset.image b Finset.univ).erase k) l
      he : Equiv (Subtype fun a => Eq (b' a) l) (Subtype fun a => Eq (b a) l) := Equ …
      ⊢ Eq (Matrix.of fun i j => M.toSquareBlockProp (fun i => Ne (b i) k) ↑i ↑j).de …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case H.inr
      α : Type u_1
      R : Type v
      inst✝⁴ : CommRing R
      inst✝³ : DecidableEq α
      inst✝² : LinearOrder α
      m : Type u_3
      M : Matrix m m R
      b : m → α
      inst✝¹ : DecidableEq m
      inst✝ : Fintype m
      hM : M.BlockTriangular b
      ih : ∀ (t : Finset α), HasSSubset.SSubset t (Finset.image b Finset.univ) → ∀ { …
      h✝ : Nonempty m
      k : α := (Finset.image b Finset.univ).max' ⋯
      ⊢ ∀ (i : m), Eq (b i) k → ∀ (j : m), Not (Eq (b j) k) → Eq (M i j) 0
    -/
  · intro i hi j hj
    /-
      case H.inr
      α : Type u_1
      R : Type v
      inst✝⁴ : CommRing R
      inst✝³ : DecidableEq α
      inst✝² : LinearOrder α
      m : Type u_3
      M : Matrix m m R
      b : m → α
      inst✝¹ : DecidableEq m
      inst✝ : Fintype m
      hM : M.BlockTriangular b
      ih : ∀ (t : Finset α), HasSSubset.SSubset t (Finset.image b Finset.univ) → ∀ { …
      h✝ : Nonempty m
      k : α := (Finset.image b Finset.univ).max' ⋯
      i : m
      hi : Eq (b i) k
      j : m
      hj : Not (Eq (b j) k)
      ⊢ Eq (M i j) 0
    -/
    apply hM
    /-
      case H.inr.a
      α : Type u_1
      R : Type v
      inst✝⁴ : CommRing R
      inst✝³ : DecidableEq α
      inst✝² : LinearOrder α
      m : Type u_3
      M : Matrix m m R
      b : m → α
      inst✝¹ : DecidableEq m
      inst✝ : Fintype m
      hM : M.BlockTriangular b
      ih : ∀ (t : Finset α), HasSSubset.SSubset t (Finset.image b Finset.univ) → ∀ { …
      h✝ : Nonempty m
      k : α := (Finset.image b Finset.univ).max' ⋯
      i : m
      hi : Eq (b i) k
      j : m
      hj : Not (Eq (b j) k)
      ⊢ LT.lt (b j) (b i)
    -/
    rw [hi]
    /-
      case H.inr.a
      α : Type u_1
      R : Type v
      inst✝⁴ : CommRing R
      inst✝³ : DecidableEq α
      inst✝² : LinearOrder α
      m : Type u_3
      M : Matrix m m R
      b : m → α
      inst✝¹ : DecidableEq m
      inst✝ : Fintype m
      hM : M.BlockTriangular b
      ih : ∀ (t : Finset α), HasSSubset.SSubset t (Finset.image b Finset.univ) → ∀ { …
      h✝ : Nonempty m
      k : α := (Finset.image b Finset.univ).max' ⋯
      i : m
      hi : Eq (b i) k
      j : m
      hj : Not (Eq (b j) k)
      ⊢ LT.lt (b j) k
    -/
    apply lt_of_le_of_ne _ hj
    /-
      α : Type u_1
      R : Type v
      inst✝⁴ : CommRing R
      inst✝³ : DecidableEq α
      inst✝² : LinearOrder α
      m : Type u_3
      M : Matrix m m R
      b : m → α
      inst✝¹ : DecidableEq m
      inst✝ : Fintype m
      hM : M.BlockTriangular b
      ih : ∀ (t : Finset α), HasSSubset.SSubset t (Finset.image b Finset.univ) → ∀ { …
      h✝ : Nonempty m
      k : α := (Finset.image b Finset.univ).max' ⋯
      i : m
      hi : Eq (b i) k
      j : m
      hj : Not (Eq (b j) k)
      ⊢ LE.le (b j) k
    -/
    exact Finset.le_max' (univ.image b) _ (mem_image_of_mem _ (mem_univ _))
    /-
      🎉 no goals
    -/


theorem BlockTriangular.det_fintype [DecidableEq α] [Fintype α] [LinearOrder α]
    (h : BlockTriangular M b) : M.det = ∏ k : α, (M.toSquareBlock b k).det := by
  /-
    α : Type u_1
    m : Type u_3
    R : Type v
    M : Matrix m m R
    b : m → α
    inst✝⁵ : CommRing R
    inst✝⁴ : DecidableEq m
    inst✝³ : Fintype m
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    inst✝ : LinearOrder α
    h : M.BlockTriangular b
    ⊢ Eq M.det (Finset.univ.prod fun k => (M.toSquareBlock b k).det)
  -/
  refine h.det.trans (prod_subset (subset_univ _) fun a _ ha => ?_)
  /-
    α : Type u_1
    m : Type u_3
    R : Type v
    M : Matrix m m R
    b : m → α
    inst✝⁵ : CommRing R
    inst✝⁴ : DecidableEq m
    inst✝³ : Fintype m
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    inst✝ : LinearOrder α
    h : M.BlockTriangular b
    a : α
    x✝ : Membership.mem Finset.univ a
    ha : Not (Membership.mem (Finset.image b Finset.univ) a)
    ⊢ Eq (M.toSquareBlock b a).det 1
  -/
  have : IsEmpty { i // b i = a } := ⟨fun i => ha <| mem_image.2 ⟨i, mem_univ _, i.2⟩⟩
  /-
    α : Type u_1
    m : Type u_3
    R : Type v
    M : Matrix m m R
    b : m → α
    inst✝⁵ : CommRing R
    inst✝⁴ : DecidableEq m
    inst✝³ : Fintype m
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    inst✝ : LinearOrder α
    h : M.BlockTriangular b
    a : α
    x✝ : Membership.mem Finset.univ a
    ha : Not (Membership.mem (Finset.image b Finset.univ) a)
    this : IsEmpty (Subtype fun i => Eq (b i) a)
    ⊢ Eq (M.toSquareBlock b a).det 1
  -/
  exact det_isEmpty
  /-
    🎉 no goals
  -/


theorem det_of_upperTriangular [LinearOrder m] (h : M.BlockTriangular id) :
    M.det = ∏ i : m, M i i := by
  /-
    m : Type u_3
    R : Type v
    M : Matrix m m R
    inst✝³ : CommRing R
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    inst✝ : LinearOrder m
    h : M.BlockTriangular id
    ⊢ Eq M.det (Finset.univ.prod fun i => M i i)
  -/
  haveI : DecidableEq R := Classical.decEq _
  /-
    m : Type u_3
    R : Type v
    M : Matrix m m R
    inst✝³ : CommRing R
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    inst✝ : LinearOrder m
    h : M.BlockTriangular id
    this : DecidableEq R
    ⊢ Eq M.det (Finset.univ.prod fun i => M i i)
  -/
  simp_rw [h.det, image_id, det_toSquareBlock_id]
  /-
    🎉 no goals
  -/


theorem det_of_lowerTriangular [LinearOrder m] (M : Matrix m m R) (h : M.BlockTriangular toDual) :
    M.det = ∏ i : m, M i i := by
  /-
    m : Type u_3
    R : Type v
    inst✝³ : CommRing R
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    inst✝ : LinearOrder m
    M : Matrix m m R
    h : M.BlockTriangular ⇑OrderDual.toDual
    ⊢ Eq M.det (Finset.univ.prod fun i => M i i)
  -/
  rw [← det_transpose]
  /-
    m : Type u_3
    R : Type v
    inst✝³ : CommRing R
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    inst✝ : LinearOrder m
    M : Matrix m m R
    h : M.BlockTriangular ⇑OrderDual.toDual
    ⊢ Eq M.transpose.det (Finset.univ.prod fun i => M i i)
  -/
  exact det_of_upperTriangular h.transpose
  /-
    🎉 no goals
  -/


theorem matrixOfPolynomials_blockTriangular {R} [Semiring R] {n : ℕ} (p : Fin n → R[X])
    (h_deg : ∀ i, (p i).natDegree ≤ i) :
    Matrix.BlockTriangular (Matrix.of (fun (i j : Fin n) => (p j).coeff i)) id :=
  fun _ j h => by
    /-
      R : Type u_8
      inst✝ : Semiring R
      n : Nat
      p : Fin n → Polynomial R
      h_deg : ∀ (i : Fin n), LE.le (p i).natDegree ↑i
      x✝ j : Fin n
      h : LT.lt (id j) (id x✝)
      ⊢ Eq (Matrix.of (fun i j => (p j).coeff ↑i) x✝ j) 0
    -/
    exact coeff_eq_zero_of_natDegree_lt <| Nat.lt_of_le_of_lt (h_deg j) h
    /-
      🎉 no goals
    -/


theorem det_matrixOfPolynomials {n : ℕ} (p : Fin n → R[X])
    (h_deg : ∀ i, (p i).natDegree = i) (h_monic : ∀ i, Monic <| p i) :
    (Matrix.of (fun (i j : Fin n) => (p j).coeff i)).det = 1 := by
  rw [Matrix.det_of_upperTriangular (Matrix.matrixOfPolynomials_blockTriangular p (fun i ↦
      Nat.le_of_eq (h_deg i)))]
  /-
    R : Type v
    inst✝ : CommRing R
    n : Nat
    p : Fin n → Polynomial R
    h_deg : ∀ (i : Fin n), Eq (p i).natDegree ↑i
    h_monic : ∀ (i : Fin n), (p i).Monic
    ⊢ Eq (Finset.univ.prod fun i => Matrix.of (fun i j => (p j).coeff ↑i) i i) 1
  -/
  convert prod_const_one with x _
  /-
    case h.e'_2.a
    R : Type v
    inst✝ : CommRing R
    n : Nat
    p : Fin n → Polynomial R
    h_deg : ∀ (i : Fin n), Eq (p i).natDegree ↑i
    h_monic : ∀ (i : Fin n), (p i).Monic
    x : Fin n
    a✝ : Membership.mem Finset.univ x
    ⊢ Eq (Matrix.of (fun i j => (p j).coeff ↑i) x x) 1
  -/
  rw [Matrix.of_apply, ← h_deg, coeff_natDegree, (h_monic x).leadingCoeff]
  /-
    🎉 no goals
  -/


theorem BlockTriangular.toBlock_inverse_mul_toBlock_eq_one [LinearOrder α] [Invertible M]
    (hM : BlockTriangular M b) (k : α) :
    ((M⁻¹.toBlock (fun i => b i < k) fun i => b i < k) *
        M.toBlock (fun i => b i < k) fun i => b i < k) =
      1 := by
  /-
    α : Type u_1
    m : Type u_3
    R : Type v
    M : Matrix m m R
    b : m → α
    inst✝⁴ : CommRing R
    inst✝³ : DecidableEq m
    inst✝² : Fintype m
    inst✝¹ : LinearOrder α
    inst✝ : Invertible M
    hM : M.BlockTriangular b
    k : α
    ⊢ Eq (HMul.hMul ((Inv.inv M).toBlock (fun i => LT.lt (b i) k) fun i => LT.lt ( …
  -/
  let p i := b i < k
  have h_sum :
    M⁻¹.toBlock p p * M.toBlock p p +
        (M⁻¹.toBlock p fun i => ¬p i) * M.toBlock (fun i => ¬p i) p =
      1 := by
    rw [← toBlock_mul_eq_add, inv_mul_of_invertible M, toBlock_one_self]
  have h_zero : M.toBlock (fun i => ¬p i) p = 0 := by
    ext i j
    simpa using hM (lt_of_lt_of_le j.2 (le_of_not_lt i.2))
  /-
    α : Type u_1
    m : Type u_3
    R : Type v
    M : Matrix m m R
    b : m → α
    inst✝⁴ : CommRing R
    inst✝³ : DecidableEq m
    inst✝² : Fintype m
    inst✝¹ : LinearOrder α
    inst✝ : Invertible M
    hM : M.BlockTriangular b
    k : α
    p : m → Prop := fun i => LT.lt (b i) k
    h_sum : Eq (HAdd.hAdd (HMul.hMul ((Inv.inv M).toBlock p p) (M.toBlock p p)) (H …
    h_zero : Eq (M.toBlock (fun i => Not (p i)) p) 0
    ⊢ Eq (HMul.hMul ((Inv.inv M).toBlock (fun i => LT.lt (b i) k) fun i => LT.lt ( …
  -/
  simpa [h_zero] using h_sum
  /-
    🎉 no goals
  -/


/-- The inverse of an upper-left subblock of a block-triangular matrix `M` is the upper-left
subblock of `M⁻¹`. -/
theorem BlockTriangular.inv_toBlock [LinearOrder α] [Invertible M] (hM : BlockTriangular M b)
    (k : α) :
    (M.toBlock (fun i => b i < k) fun i => b i < k)⁻¹ =
      M⁻¹.toBlock (fun i => b i < k) fun i => b i < k :=
  inv_eq_left_inv <| hM.toBlock_inverse_mul_toBlock_eq_one k


/-- An upper-left subblock of an invertible block-triangular matrix is invertible. -/
def BlockTriangular.invertibleToBlock [LinearOrder α] [Invertible M] (hM : BlockTriangular M b)
    (k : α) : Invertible (M.toBlock (fun i => b i < k) fun i => b i < k) :=
  invertibleOfLeftInverse _ ((⅟ M).toBlock (fun i => b i < k) fun i => b i < k) <| by
    /-
      α : Type u_1
      β : Type u_2
      m : Type u_3
      n : Type u_4
      o : Type u_5
      m' : α → Type u_6
      n' : α → Type u_7
      R : Type v
      M N : Matrix m m R
      b : m → α
      inst✝⁶ : CommRing R
      inst✝⁵ : DecidableEq m
      inst✝⁴ : Fintype m
      inst✝³ : DecidableEq n
      inst✝² : Fintype n
      inst✝¹ : LinearOrder α
      inst✝ : Invertible M
      hM : M.BlockTriangular b
      k : α
      ⊢ Eq (HMul.hMul ((Invertible.invOf M).toBlock (fun i => LT.lt (b i) k) fun i = …
    -/
    simpa only [invOf_eq_nonsing_inv] using hM.toBlock_inverse_mul_toBlock_eq_one k
    /-
      🎉 no goals
    -/


/-- A lower-left subblock of the inverse of a block-triangular matrix is zero. This is a first step
towards `BlockTriangular.inv_toBlock` below. -/
theorem toBlock_inverse_eq_zero [LinearOrder α] [Invertible M] (hM : BlockTriangular M b) (k : α) :
    (M⁻¹.toBlock (fun i => k ≤ b i) fun i => b i < k) = 0 := by
  /-
    α : Type u_1
    m : Type u_3
    R : Type v
    M : Matrix m m R
    b : m → α
    inst✝⁴ : CommRing R
    inst✝³ : DecidableEq m
    inst✝² : Fintype m
    inst✝¹ : LinearOrder α
    inst✝ : Invertible M
    hM : M.BlockTriangular b
    k : α
    ⊢ Eq ((Inv.inv M).toBlock (fun i => LE.le k (b i)) fun i => LT.lt (b i) k) 0
  -/
  let p i := b i < k
  /-
    α : Type u_1
    m : Type u_3
    R : Type v
    M : Matrix m m R
    b : m → α
    inst✝⁴ : CommRing R
    inst✝³ : DecidableEq m
    inst✝² : Fintype m
    inst✝¹ : LinearOrder α
    inst✝ : Invertible M
    hM : M.BlockTriangular b
    k : α
    p : m → Prop := fun i => LT.lt (b i) k
    ⊢ Eq ((Inv.inv M).toBlock (fun i => LE.le k (b i)) fun i => LT.lt (b i) k) 0
  -/
  let q i := ¬b i < k
  have h_sum : M⁻¹.toBlock q p * M.toBlock p p + M⁻¹.toBlock q q * M.toBlock q p = 0 := by
    rw [← toBlock_mul_eq_add, inv_mul_of_invertible M, toBlock_one_disjoint]
    rw [disjoint_iff_inf_le]
    exact fun i h => h.1 h.2
  have h_zero : M.toBlock q p = 0 := by
    ext i j
    simpa using hM (lt_of_lt_of_le j.2 <| le_of_not_lt i.2)
  /-
    α : Type u_1
    m : Type u_3
    R : Type v
    M : Matrix m m R
    b : m → α
    inst✝⁴ : CommRing R
    inst✝³ : DecidableEq m
    inst✝² : Fintype m
    inst✝¹ : LinearOrder α
    inst✝ : Invertible M
    hM : M.BlockTriangular b
    k : α
    p : m → Prop := fun i => LT.lt (b i) k
    q : m → Prop := fun i => Not (LT.lt (b i) k)
    h_sum : Eq (HAdd.hAdd (HMul.hMul ((Inv.inv M).toBlock q p) (M.toBlock p p)) (H …
    h_zero : Eq (M.toBlock q p) 0
    ⊢ Eq ((Inv.inv M).toBlock (fun i => LE.le k (b i)) fun i => LT.lt (b i) k) 0
  -/
  have h_mul_eq_zero : M⁻¹.toBlock q p * M.toBlock p p = 0 := by simpa [h_zero] using h_sum
  /-
    α : Type u_1
    m : Type u_3
    R : Type v
    M : Matrix m m R
    b : m → α
    inst✝⁴ : CommRing R
    inst✝³ : DecidableEq m
    inst✝² : Fintype m
    inst✝¹ : LinearOrder α
    inst✝ : Invertible M
    hM : M.BlockTriangular b
    k : α
    p : m → Prop := fun i => LT.lt (b i) k
    q : m → Prop := fun i => Not (LT.lt (b i) k)
    h_sum : Eq (HAdd.hAdd (HMul.hMul ((Inv.inv M).toBlock q p) (M.toBlock p p)) (H …
    h_zero : Eq (M.toBlock q p) 0
    h_mul_eq_zero : Eq (HMul.hMul ((Inv.inv M).toBlock q p) (M.toBlock p p)) 0
    ⊢ Eq ((Inv.inv M).toBlock (fun i => LE.le k (b i)) fun i => LT.lt (b i) k) 0
  -/
  haveI : Invertible (M.toBlock p p) := hM.invertibleToBlock k
  have : (fun i => k ≤ b i) = q := by
    ext
    exact not_lt.symm
  rw [this, ← Matrix.zero_mul (M.toBlock p p)⁻¹, ← h_mul_eq_zero,
    mul_inv_cancel_right_of_invertible]


/-- The inverse of a block-triangular matrix is block-triangular. -/
theorem blockTriangular_inv_of_blockTriangular [LinearOrder α] [Invertible M]
    (hM : BlockTriangular M b) : BlockTriangular M⁻¹ b := by
  /-
    α : Type u_1
    m : Type u_3
    R : Type v
    M : Matrix m m R
    b : m → α
    inst✝⁴ : CommRing R
    inst✝³ : DecidableEq m
    inst✝² : Fintype m
    inst✝¹ : LinearOrder α
    inst✝ : Invertible M
    hM : M.BlockTriangular b
    ⊢ (Inv.inv M).BlockTriangular b
  -/
  induction' hs : univ.image b using Finset.strongInduction with s ih generalizing m
  /-
    case H
    α : Type u_1
    R : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LinearOrder α
    s : Finset α
    ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {m : Type u_3} {M : Matrix m …
    m : Type u_3
    M : Matrix m m R
    b : m → α
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    inst✝ : Invertible M
    hM : M.BlockTriangular b
    hs : Eq (Finset.image b Finset.univ) s
    ⊢ (Inv.inv M).BlockTriangular b
  -/
  subst hs
  /-
    case H
    α : Type u_1
    R : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LinearOrder α
    m : Type u_3
    M : Matrix m m R
    b : m → α
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    inst✝ : Invertible M
    hM : M.BlockTriangular b
    ih : ∀ (t : Finset α), HasSSubset.SSubset t (Finset.image b Finset.univ) → ∀ { …
    ⊢ (Inv.inv M).BlockTriangular b
  -/
  intro i j hij
  /-
    case H
    α : Type u_1
    R : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LinearOrder α
    m : Type u_3
    M : Matrix m m R
    b : m → α
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    inst✝ : Invertible M
    hM : M.BlockTriangular b
    ih : ∀ (t : Finset α), HasSSubset.SSubset t (Finset.image b Finset.univ) → ∀ { …
    i j : m
    hij : LT.lt (b j) (b i)
    ⊢ Eq (Inv.inv M i j) 0
  -/
  haveI : Inhabited m := ⟨i⟩
  /-
    case H
    α : Type u_1
    R : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LinearOrder α
    m : Type u_3
    M : Matrix m m R
    b : m → α
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    inst✝ : Invertible M
    hM : M.BlockTriangular b
    ih : ∀ (t : Finset α), HasSSubset.SSubset t (Finset.image b Finset.univ) → ∀ { …
    i j : m
    hij : LT.lt (b j) (b i)
    this : Inhabited m
    ⊢ Eq (Inv.inv M i j) 0
  -/
  let k := (univ.image b).max' (univ_nonempty.image _)
  /-
    case H
    α : Type u_1
    R : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LinearOrder α
    m : Type u_3
    M : Matrix m m R
    b : m → α
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    inst✝ : Invertible M
    hM : M.BlockTriangular b
    ih : ∀ (t : Finset α), HasSSubset.SSubset t (Finset.image b Finset.univ) → ∀ { …
    i j : m
    hij : LT.lt (b j) (b i)
    this : Inhabited m
    k : α := (Finset.image b Finset.univ).max' ⋯
    ⊢ Eq (Inv.inv M i j) 0
  -/
  let b' := fun i : { a // b a < k } => b ↑i
  /-
    case H
    α : Type u_1
    R : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LinearOrder α
    m : Type u_3
    M : Matrix m m R
    b : m → α
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    inst✝ : Invertible M
    hM : M.BlockTriangular b
    ih : ∀ (t : Finset α), HasSSubset.SSubset t (Finset.image b Finset.univ) → ∀ { …
    i j : m
    hij : LT.lt (b j) (b i)
    this : Inhabited m
    k : α := (Finset.image b Finset.univ).max' ⋯
    b' : (Subtype fun a => LT.lt (b a) k) → α := fun i => b ↑i
    ⊢ Eq (Inv.inv M i j) 0
  -/
  let A := M.toBlock (fun i => b i < k) fun j => b j < k
  /-
    case H
    α : Type u_1
    R : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LinearOrder α
    m : Type u_3
    M : Matrix m m R
    b : m → α
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    inst✝ : Invertible M
    hM : M.BlockTriangular b
    ih : ∀ (t : Finset α), HasSSubset.SSubset t (Finset.image b Finset.univ) → ∀ { …
    i j : m
    hij : LT.lt (b j) (b i)
    this : Inhabited m
    k : α := (Finset.image b Finset.univ).max' ⋯
    b' : (Subtype fun a => LT.lt (b a) k) → α := fun i => b ↑i
    A : Matrix (Subtype fun a => LT.lt (b a) k) (Subtype fun a => LT.lt (b a) k) R …
    ⊢ Eq (Inv.inv M i j) 0
  -/
  obtain hbi | hi : b i = k ∨ _ := (le_max' _ (b i) <| mem_image_of_mem _ <| mem_univ _).eq_or_lt
  · have : M⁻¹.toBlock (fun i => k ≤ b i) (fun i => b i < k) ⟨i, hbi.ge⟩ ⟨j, hbi ▸ hij⟩ = 0 := by
      simp only [toBlock_inverse_eq_zero hM k, Matrix.zero_apply]
    /-
      case H.inl
      α : Type u_1
      R : Type v
      inst✝⁴ : CommRing R
      inst✝³ : LinearOrder α
      m : Type u_3
      M : Matrix m m R
      b : m → α
      inst✝² : DecidableEq m
      inst✝¹ : Fintype m
      inst✝ : Invertible M
      hM : M.BlockTriangular b
      ih : ∀ (t : Finset α), HasSSubset.SSubset t (Finset.image b Finset.univ) → ∀ { …
      i j : m
      hij : LT.lt (b j) (b i)
      this✝ : Inhabited m
      k : α := (Finset.image b Finset.univ).max' ⋯
      b' : (Subtype fun a => LT.lt (b a) k) → α := fun i => b ↑i
      A : Matrix (Subtype fun a => LT.lt (b a) k) (Subtype fun a => LT.lt (b a) k) R …
      hbi : Eq (b i) k
      this : Eq ((Inv.inv M).toBlock (fun i => LE.le k (b i)) (fun i => LT.lt (b i)  …
      ⊢ Eq (Inv.inv M i j) 0
    -/
    simp [this.symm]
    /-
      🎉 no goals
    -/
  /-
    case H.inr
    α : Type u_1
    R : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LinearOrder α
    m : Type u_3
    M : Matrix m m R
    b : m → α
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    inst✝ : Invertible M
    hM : M.BlockTriangular b
    ih : ∀ (t : Finset α), HasSSubset.SSubset t (Finset.image b Finset.univ) → ∀ { …
    i j : m
    hij : LT.lt (b j) (b i)
    this : Inhabited m
    k : α := (Finset.image b Finset.univ).max' ⋯
    b' : (Subtype fun a => LT.lt (b a) k) → α := fun i => b ↑i
    A : Matrix (Subtype fun a => LT.lt (b a) k) (Subtype fun a => LT.lt (b a) k) R …
    hi : LT.lt (b i) ((Finset.image b Finset.univ).max' ⋯)
    ⊢ Eq (Inv.inv M i j) 0
  -/
  haveI : Invertible A := hM.invertibleToBlock _
  /-
    case H.inr
    α : Type u_1
    R : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LinearOrder α
    m : Type u_3
    M : Matrix m m R
    b : m → α
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    inst✝ : Invertible M
    hM : M.BlockTriangular b
    ih : ∀ (t : Finset α), HasSSubset.SSubset t (Finset.image b Finset.univ) → ∀ { …
    i j : m
    hij : LT.lt (b j) (b i)
    this✝ : Inhabited m
    k : α := (Finset.image b Finset.univ).max' ⋯
    b' : (Subtype fun a => LT.lt (b a) k) → α := fun i => b ↑i
    A : Matrix (Subtype fun a => LT.lt (b a) k) (Subtype fun a => LT.lt (b a) k) R …
    hi : LT.lt (b i) ((Finset.image b Finset.univ).max' ⋯)
    this : Invertible A
    ⊢ Eq (Inv.inv M i j) 0
  -/
  have hA : A.BlockTriangular b' := hM.submatrix
  have hb' : image b' univ ⊂ image b univ := by
    convert image_subtype_univ_ssubset_image_univ k b _ (fun a => a < k) (lt_irrefl _)
    convert max'_mem (α := α) _ _
  /-
    case H.inr
    α : Type u_1
    R : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LinearOrder α
    m : Type u_3
    M : Matrix m m R
    b : m → α
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    inst✝ : Invertible M
    hM : M.BlockTriangular b
    ih : ∀ (t : Finset α), HasSSubset.SSubset t (Finset.image b Finset.univ) → ∀ { …
    i j : m
    hij : LT.lt (b j) (b i)
    this✝ : Inhabited m
    k : α := (Finset.image b Finset.univ).max' ⋯
    b' : (Subtype fun a => LT.lt (b a) k) → α := fun i => b ↑i
    A : Matrix (Subtype fun a => LT.lt (b a) k) (Subtype fun a => LT.lt (b a) k) R …
    hi : LT.lt (b i) ((Finset.image b Finset.univ).max' ⋯)
    this : Invertible A
    hA : A.BlockTriangular b'
    hb' : HasSSubset.SSubset (Finset.image b' Finset.univ) (Finset.image b Finset. …
    ⊢ Eq (Inv.inv M i j) 0
  -/
  have hij' : b' ⟨j, hij.trans hi⟩ < b' ⟨i, hi⟩ := by simp_rw [b', hij]
  /-
    case H.inr
    α : Type u_1
    R : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LinearOrder α
    m : Type u_3
    M : Matrix m m R
    b : m → α
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    inst✝ : Invertible M
    hM : M.BlockTriangular b
    ih : ∀ (t : Finset α), HasSSubset.SSubset t (Finset.image b Finset.univ) → ∀ { …
    i j : m
    hij : LT.lt (b j) (b i)
    this✝ : Inhabited m
    k : α := (Finset.image b Finset.univ).max' ⋯
    b' : (Subtype fun a => LT.lt (b a) k) → α := fun i => b ↑i
    A : Matrix (Subtype fun a => LT.lt (b a) k) (Subtype fun a => LT.lt (b a) k) R …
    hi : LT.lt (b i) ((Finset.image b Finset.univ).max' ⋯)
    this : Invertible A
    hA : A.BlockTriangular b'
    hb' : HasSSubset.SSubset (Finset.image b' Finset.univ) (Finset.image b Finset. …
    hij' : LT.lt (b' ⟨j, ⋯⟩) (b' ⟨i, hi⟩)
    ⊢ Eq (Inv.inv M i j) 0
  -/
  simp [A, hM.inv_toBlock k, (ih (image b' univ) hb' hA rfl hij').symm]
  /-
    🎉 no goals
  -/


