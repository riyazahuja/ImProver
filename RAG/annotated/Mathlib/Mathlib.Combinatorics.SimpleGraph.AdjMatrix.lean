/-- `A : Matrix V V α` is qualified as an "adjacency matrix" if
    (1) every entry of `A` is `0` or `1`,
    (2) `A` is symmetric,
    (3) every diagonal entry of `A` is `0`. -/
structure IsAdjMatrix [Zero α] [One α] (A : Matrix V V α) : Prop where
  zero_or_one : ∀ i j, A i j = 0 ∨ A i j = 1 := by aesop
  symm : A.IsSymm := by aesop
  apply_diag : ∀ i, A i i = 0 := by aesop


@[simp]
theorem apply_diag_ne [MulZeroOneClass α] [Nontrivial α] (h : IsAdjMatrix A) (i : V) :
                     /-
                       V : Type u_1
                       α : Type u_2
                       A : Matrix V V α
                       inst✝¹ : MulZeroOneClass α
                       inst✝ : Nontrivial α
                       h : A.IsAdjMatrix
                       i : V
                       ⊢ Not (Eq (A i i) 1)
                     -/
    ¬A i i = 1 := by simp [h.apply_diag i]
                     /-
                       🎉 no goals
                     -/


@[simp]
theorem apply_ne_one_iff [MulZeroOneClass α] [Nontrivial α] (h : IsAdjMatrix A) (i j : V) :
                                 /-
                                   V : Type u_1
                                   α : Type u_2
                                   A : Matrix V V α
                                   inst✝¹ : MulZeroOneClass α
                                   inst✝ : Nontrivial α
                                   h : A.IsAdjMatrix
                                   i j : V
                                   ⊢ Iff (Not (Eq (A i j) 1)) (Eq (A i j) 0)
                                 -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
    ¬A i j = 1 ↔ A i j = 0 := by obtain h | h := h.zero_or_one i j <;> simp [h]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp]
theorem apply_ne_zero_iff [MulZeroOneClass α] [Nontrivial α] (h : IsAdjMatrix A) (i j : V) :
                                 /-
                                   V : Type u_1
                                   α : Type u_2
                                   A : Matrix V V α
                                   inst✝¹ : MulZeroOneClass α
                                   inst✝ : Nontrivial α
                                   h : A.IsAdjMatrix
                                   i j : V
                                   ⊢ Iff (Not (Eq (A i j) 0)) (Eq (A i j) 1)
                                 -/
    ¬A i j = 0 ↔ A i j = 1 := by rw [← apply_ne_one_iff h, Classical.not_not]
                                 /-
                                   🎉 no goals
                                 -/


/-- For `A : Matrix V V α` and `h : IsAdjMatrix A`,
    `h.toGraph` is the simple graph whose adjacency matrix is `A`. -/
@[simps]
def toGraph [MulZeroOneClass α] [Nontrivial α] (h : IsAdjMatrix A) : SimpleGraph V where
  Adj i j := A i j = 1
                     /-
                       V : Type u_1
                       α : Type u_2
                       A : Matrix V V α
                       inst✝¹ : MulZeroOneClass α
                       inst✝ : Nontrivial α
                       h : A.IsAdjMatrix
                       i j : V
                       hij : (fun i j => Eq (A i j) 1) i j
                       ⊢ (fun i j => Eq (A i j) 1) j i
                     -/
  symm i j hij := by simp only; rwa [h.symm.apply i j]
                                /-
                                  🎉 no goals
                                -/
                   /-
                     V : Type u_1
                     α : Type u_2
                     A : Matrix V V α
                     inst✝¹ : MulZeroOneClass α
                     inst✝ : Nontrivial α
                     h : A.IsAdjMatrix
                     i : V
                     ⊢ Not ((fun i j => Eq (A i j) 1) i i)
                   -/
  loopless i := by simp [h]
                   /-
                     🎉 no goals
                   -/


instance [MulZeroOneClass α] [Nontrivial α] [DecidableEq α] (h : IsAdjMatrix A) :
    DecidableRel h.toGraph.Adj := by
  /-
    V : Type u_1
    α : Type u_2
    A : Matrix V V α
    inst✝² : MulZeroOneClass α
    inst✝¹ : Nontrivial α
    inst✝ : DecidableEq α
    h : A.IsAdjMatrix
    ⊢ DecidableRel h.toGraph.Adj
  -/
  simp only [toGraph]
  /-
    V : Type u_1
    α : Type u_2
    A : Matrix V V α
    inst✝² : MulZeroOneClass α
    inst✝¹ : Nontrivial α
    inst✝ : DecidableEq α
    h : A.IsAdjMatrix
    ⊢ DecidableRel fun i j => Eq (A i j) 1
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- For `A : Matrix V V α`, `A.compl` is supposed to be the adjacency matrix of
    the complement graph of the graph induced by `A.adjMatrix`. -/
def compl [Zero α] [One α] [DecidableEq α] [DecidableEq V] (A : Matrix V V α) : Matrix V V α :=
  fun i j => ite (i = j) 0 (ite (A i j = 0) 1 0)


@[simp]
                                                                          /-
                                                                            V : Type u_1
                                                                            α : Type u_2
                                                                            inst✝³ : DecidableEq α
                                                                            inst✝² : DecidableEq V
                                                                            A : Matrix V V α
                                                                            inst✝¹ : Zero α
                                                                            inst✝ : One α
                                                                            i : V
                                                                            ⊢ Eq (A.compl i i) 0
                                                                          -/
theorem compl_apply_diag [Zero α] [One α] (i : V) : A.compl i i = 0 := by simp [compl]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[simp]
theorem compl_apply [Zero α] [One α] (i j : V) : A.compl i j = 0 ∨ A.compl i j = 1 := by
  /-
    V : Type u_1
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : DecidableEq V
    A : Matrix V V α
    inst✝¹ : Zero α
    inst✝ : One α
    i j : V
    ⊢ Or (Eq (A.compl i j) 0) (Eq (A.compl i j) 1)
  -/
  unfold compl
  /-
    V : Type u_1
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : DecidableEq V
    A : Matrix V V α
    inst✝¹ : Zero α
    inst✝ : One α
    i j : V
    ⊢ Or (Eq (ite (Eq i j) 0 (ite (Eq (A i j) 0) 1 0)) 0) (Eq (ite (Eq i j) 0 (ite …
  -/
                /-
                  🎉 no goals
                -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp
                /-
                  🎉 no goals
                -/


@[simp]
theorem isSymm_compl [Zero α] [One α] (h : A.IsSymm) : A.compl.IsSymm := by
  /-
    V : Type u_1
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : DecidableEq V
    A : Matrix V V α
    inst✝¹ : Zero α
    inst✝ : One α
    h : A.IsSymm
    ⊢ A.compl.IsSymm
  -/
  ext
  /-
    case a
    V : Type u_1
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : DecidableEq V
    A : Matrix V V α
    inst✝¹ : Zero α
    inst✝ : One α
    h : A.IsSymm
    i✝ j✝ : V
    ⊢ Eq (A.compl.transpose i✝ j✝) (A.compl i✝ j✝)
  -/
  simp [compl, h.apply, eq_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem isAdjMatrix_compl [Zero α] [One α] (h : A.IsSymm) : IsAdjMatrix A.compl :=
               /-
                 V : Type u_1
                 α : Type u_2
                 inst✝³ : DecidableEq α
                 inst✝² : DecidableEq V
                 A : Matrix V V α
                 inst✝¹ : Zero α
                 inst✝ : One α
                 h : A.IsSymm
                 ⊢ A.compl.IsSymm
               -/
  { symm := by simp [h] }
               /-
                 🎉 no goals
               -/


@[simp]
theorem compl [Zero α] [One α] (h : IsAdjMatrix A) : IsAdjMatrix A.compl :=
  isAdjMatrix_compl A h.symm


theorem toGraph_compl_eq [MulZeroOneClass α] [Nontrivial α] (h : IsAdjMatrix A) :
    h.compl.toGraph = h.toGraphᶜ := by
  /-
    V : Type u_1
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : DecidableEq V
    A : Matrix V V α
    inst✝¹ : MulZeroOneClass α
    inst✝ : Nontrivial α
    h : A.IsAdjMatrix
    ⊢ Eq ⋯.toGraph (HasCompl.compl h.toGraph)
  -/
  ext v w
  /-
    case Adj.h.h.a
    V : Type u_1
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : DecidableEq V
    A : Matrix V V α
    inst✝¹ : MulZeroOneClass α
    inst✝ : Nontrivial α
    h : A.IsAdjMatrix
    v w : V
    ⊢ Iff (⋯.toGraph.Adj v w) ((HasCompl.compl h.toGraph).Adj v w)
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
  cases' h.zero_or_one v w with h h <;> by_cases hvw : v = w <;> simp [Matrix.compl, h, hvw]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- `adjMatrix G α` is the matrix `A` such that `A i j = (1 : α)` if `i` and `j` are
  adjacent in the simple graph `G`, and otherwise `A i j = 0`. -/
def adjMatrix [Zero α] [One α] : Matrix V V α :=
  of fun i j => if G.Adj i j then (1 : α) else 0


@[simp]
theorem adjMatrix_apply (v w : V) [Zero α] [One α] :
    G.adjMatrix α v w = if G.Adj v w then 1 else 0 :=
  rfl


@[simp]
theorem transpose_adjMatrix [Zero α] [One α] : (G.adjMatrix α)ᵀ = G.adjMatrix α := by
  /-
    V : Type u_1
    α : Type u_2
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : Zero α
    inst✝ : One α
    ⊢ Eq (SimpleGraph.adjMatrix α G).transpose (SimpleGraph.adjMatrix α G)
  -/
  ext
  /-
    case a
    V : Type u_1
    α : Type u_2
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : Zero α
    inst✝ : One α
    i✝ j✝ : V
    ⊢ Eq ((SimpleGraph.adjMatrix α G).transpose i✝ j✝) (SimpleGraph.adjMatrix α G  …
  -/
  simp [adj_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem isSymm_adjMatrix [Zero α] [One α] : (G.adjMatrix α).IsSymm :=
  transpose_adjMatrix G


/-- The adjacency matrix of `G` is an adjacency matrix. -/
@[simp]
theorem isAdjMatrix_adjMatrix [Zero α] [One α] : (G.adjMatrix α).IsAdjMatrix :=
                                 /-
                                   V : Type u_1
                                   α : Type u_2
                                   G : SimpleGraph V
                                   inst✝² : DecidableRel G.Adj
                                   inst✝¹ : Zero α
                                   inst✝ : One α
                                   i j : V
                                   ⊢ Or (Eq (SimpleGraph.adjMatrix α G i j) 0) (Eq (SimpleGraph.adjMatrix α G i j …
                                 -/
                                                            /-
                                                              🎉 no goals
                                                            -/
  { zero_or_one := fun i j => by by_cases h : G.Adj i j <;> simp [h] }
                                                            /-
                                                              🎉 no goals
                                                            -/


/-- The graph induced by the adjacency matrix of `G` is `G` itself. -/
theorem toGraph_adjMatrix_eq [MulZeroOneClass α] [Nontrivial α] :
    (G.isAdjMatrix_adjMatrix α).toGraph = G := by
  /-
    V : Type u_1
    α : Type u_2
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : MulZeroOneClass α
    inst✝ : Nontrivial α
    ⊢ Eq ⋯.toGraph G
  -/
  ext
  /-
    case Adj.h.h.a
    V : Type u_1
    α : Type u_2
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : MulZeroOneClass α
    inst✝ : Nontrivial α
    x✝¹ x✝ : V
    ⊢ Iff (⋯.toGraph.Adj x✝¹ x✝) (G.Adj x✝¹ x✝)
  -/
  simp only [IsAdjMatrix.toGraph_adj, adjMatrix_apply, ite_eq_left_iff, zero_ne_one]
  /-
    case Adj.h.h.a
    V : Type u_1
    α : Type u_2
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : MulZeroOneClass α
    inst✝ : Nontrivial α
    x✝¹ x✝ : V
    ⊢ Iff (Not (G.Adj x✝¹ x✝) → False) (G.Adj x✝¹ x✝)
  -/
  apply Classical.not_not
  /-
    🎉 no goals
  -/


/-- The sum of the identity, the adjacency matrix, and its complement is the all-ones matrix. -/
theorem one_add_adjMatrix_add_compl_adjMatrix_eq_allOnes [DecidableEq V] [DecidableEq α]
    [NonAssocSemiring α] : 1 + G.adjMatrix α + (G.adjMatrix α).compl = Matrix.of fun _ _ ↦ 1 := by
  /-
    V : Type u_1
    α : Type u_2
    G : SimpleGraph V
    inst✝³ : DecidableRel G.Adj
    inst✝² : DecidableEq V
    inst✝¹ : DecidableEq α
    inst✝ : NonAssocSemiring α
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd 1 (SimpleGraph.adjMatrix α G)) (SimpleGraph.adjMatr …
  -/
  ext i j
  /-
    case a
    V : Type u_1
    α : Type u_2
    G : SimpleGraph V
    inst✝³ : DecidableRel G.Adj
    inst✝² : DecidableEq V
    inst✝¹ : DecidableEq α
    inst✝ : NonAssocSemiring α
    i j : V
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd 1 (SimpleGraph.adjMatrix α G)) (SimpleGraph.adjMatr …
  -/
  unfold Matrix.compl
  /-
    case a
    V : Type u_1
    α : Type u_2
    G : SimpleGraph V
    inst✝³ : DecidableRel G.Adj
    inst✝² : DecidableEq V
    inst✝¹ : DecidableEq α
    inst✝ : NonAssocSemiring α
    i j : V
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd 1 (SimpleGraph.adjMatrix α G)) (fun i j => ite (Eq  …
  -/
  rw [of_apply, add_apply, adjMatrix_apply, add_apply, adjMatrix_apply, one_apply]
  /-
    case a
    V : Type u_1
    α : Type u_2
    G : SimpleGraph V
    inst✝³ : DecidableRel G.Adj
    inst✝² : DecidableEq V
    inst✝¹ : DecidableEq α
    inst✝ : NonAssocSemiring α
    i j : V
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (ite (Eq i j) 1 0) (ite (G.Adj i j) 1 0)) (ite (Eq  …
  -/
  by_cases h : G.Adj i j
    /-
      case pos
      V : Type u_1
      α : Type u_2
      G : SimpleGraph V
      inst✝³ : DecidableRel G.Adj
      inst✝² : DecidableEq V
      inst✝¹ : DecidableEq α
      inst✝ : NonAssocSemiring α
      i j : V
      h : G.Adj i j
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (ite (Eq i j) 1 0) (ite (G.Adj i j) 1 0)) (ite (Eq  …
    -/
  · aesop
    /-
      🎉 no goals
    -/
    /-
      case neg
      V : Type u_1
      α : Type u_2
      G : SimpleGraph V
      inst✝³ : DecidableRel G.Adj
      inst✝² : DecidableEq V
      inst✝¹ : DecidableEq α
      inst✝ : NonAssocSemiring α
      i j : V
      h : Not (G.Adj i j)
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (ite (Eq i j) 1 0) (ite (G.Adj i j) 1 0)) (ite (Eq  …
    -/
                  /-
                    🎉 no goals
                  -/
                  /-
                    🎉 no goals
                  -/
  · split_ifs <;> simp_all
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem adjMatrix_dotProduct [NonAssocSemiring α] (v : V) (vec : V → α) :
    dotProduct (G.adjMatrix α v) vec = ∑ u ∈ G.neighborFinset v, vec u := by
  /-
    V : Type u_1
    α : Type u_2
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : Fintype V
    inst✝ : NonAssocSemiring α
    v : V
    vec : V → α
    ⊢ Eq (dotProduct (SimpleGraph.adjMatrix α G v) vec) ((G.neighborFinset v).sum  …
  -/
  simp [neighborFinset_eq_filter, dotProduct, sum_filter]
  /-
    🎉 no goals
  -/


@[simp]
theorem dotProduct_adjMatrix [NonAssocSemiring α] (v : V) (vec : V → α) :
    dotProduct vec (G.adjMatrix α v) = ∑ u ∈ G.neighborFinset v, vec u := by
  /-
    V : Type u_1
    α : Type u_2
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : Fintype V
    inst✝ : NonAssocSemiring α
    v : V
    vec : V → α
    ⊢ Eq (dotProduct vec (SimpleGraph.adjMatrix α G v)) ((G.neighborFinset v).sum  …
  -/
  simp [neighborFinset_eq_filter, dotProduct, sum_filter, Finset.sum_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem adjMatrix_mulVec_apply [NonAssocSemiring α] (v : V) (vec : V → α) :
    (G.adjMatrix α *ᵥ vec) v = ∑ u ∈ G.neighborFinset v, vec u := by
  /-
    V : Type u_1
    α : Type u_2
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : Fintype V
    inst✝ : NonAssocSemiring α
    v : V
    vec : V → α
    ⊢ Eq ((SimpleGraph.adjMatrix α G).mulVec vec v) ((G.neighborFinset v).sum fun  …
  -/
  rw [mulVec, adjMatrix_dotProduct]
  /-
    🎉 no goals
  -/


@[simp]
theorem adjMatrix_vecMul_apply [NonAssocSemiring α] (v : V) (vec : V → α) :
    (vec ᵥ* G.adjMatrix α) v = ∑ u ∈ G.neighborFinset v, vec u := by
  /-
    V : Type u_1
    α : Type u_2
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : Fintype V
    inst✝ : NonAssocSemiring α
    v : V
    vec : V → α
    ⊢ Eq (Matrix.vecMul vec (SimpleGraph.adjMatrix α G) v) ((G.neighborFinset v).s …
  -/
  simp only [← dotProduct_adjMatrix, vecMul]
  /-
    V : Type u_1
    α : Type u_2
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : Fintype V
    inst✝ : NonAssocSemiring α
    v : V
    vec : V → α
    ⊢ Eq (dotProduct vec fun i => SimpleGraph.adjMatrix α G i v) (dotProduct vec ( …
  -/
  refine congr rfl ?_; ext x
  /-
    case h
    V : Type u_1
    α : Type u_2
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : Fintype V
    inst✝ : NonAssocSemiring α
    v : V
    vec : V → α
    x : V
    ⊢ Eq (SimpleGraph.adjMatrix α G x v) (SimpleGraph.adjMatrix α G v x)
  -/
  rw [← transpose_apply (adjMatrix α G) x v, transpose_adjMatrix]
  /-
    🎉 no goals
  -/


@[simp]
theorem adjMatrix_mul_apply [NonAssocSemiring α] (M : Matrix V V α) (v w : V) :
    (G.adjMatrix α * M) v w = ∑ u ∈ G.neighborFinset v, M u w := by
  /-
    V : Type u_1
    α : Type u_2
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : Fintype V
    inst✝ : NonAssocSemiring α
    M : Matrix V V α
    v w : V
    ⊢ Eq (HMul.hMul (SimpleGraph.adjMatrix α G) M v w) ((G.neighborFinset v).sum f …
  -/
  simp [mul_apply, neighborFinset_eq_filter, sum_filter]
  /-
    🎉 no goals
  -/


@[simp]
theorem mul_adjMatrix_apply [NonAssocSemiring α] (M : Matrix V V α) (v w : V) :
    (M * G.adjMatrix α) v w = ∑ u ∈ G.neighborFinset w, M v u := by
  /-
    V : Type u_1
    α : Type u_2
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : Fintype V
    inst✝ : NonAssocSemiring α
    M : Matrix V V α
    v w : V
    ⊢ Eq (HMul.hMul M (SimpleGraph.adjMatrix α G) v w) ((G.neighborFinset w).sum f …
  -/
  simp [mul_apply, neighborFinset_eq_filter, sum_filter, adj_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem trace_adjMatrix [AddCommMonoid α] [One α] : Matrix.trace (G.adjMatrix α) = 0 := by
  /-
    V : Type u_1
    α : Type u_2
    G : SimpleGraph V
    inst✝³ : DecidableRel G.Adj
    inst✝² : Fintype V
    inst✝¹ : AddCommMonoid α
    inst✝ : One α
    ⊢ Eq (SimpleGraph.adjMatrix α G).trace 0
  -/
  simp [Matrix.trace]
  /-
    🎉 no goals
  -/


theorem adjMatrix_mul_self_apply_self [NonAssocSemiring α] (i : V) :
                                                           /-
                                                             V : Type u_1
                                                             α : Type u_2
                                                             G : SimpleGraph V
                                                             inst✝² : DecidableRel G.Adj
                                                             inst✝¹ : Fintype V
                                                             inst✝ : NonAssocSemiring α
                                                             i : V
                                                             ⊢ Eq (HMul.hMul (SimpleGraph.adjMatrix α G) (SimpleGraph.adjMatrix α G) i i) ↑ …
                                                           -/
    (G.adjMatrix α * G.adjMatrix α) i i = degree G i := by simp [filter_true_of_mem]
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem adjMatrix_mulVec_const_apply [NonAssocSemiring α] {a : α} {v : V} :
                                                                   /-
                                                                     V : Type u_1
                                                                     α : Type u_2
                                                                     G : SimpleGraph V
                                                                     inst✝² : DecidableRel G.Adj
                                                                     inst✝¹ : Fintype V
                                                                     inst✝ : NonAssocSemiring α
                                                                     a : α
                                                                     v : V
                                                                     ⊢ Eq ((SimpleGraph.adjMatrix α G).mulVec (Function.const V a) v) (HMul.hMul (↑ …
                                                                   -/
    (G.adjMatrix α *ᵥ Function.const _ a) v = G.degree v * a := by simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


theorem adjMatrix_mulVec_const_apply_of_regular [NonAssocSemiring α] {d : ℕ} {a : α}
    (hd : G.IsRegularOfDegree d) {v : V} : (G.adjMatrix α *ᵥ Function.const _ a) v = d * a := by
  /-
    V : Type u_1
    α : Type u_2
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : Fintype V
    inst✝ : NonAssocSemiring α
    d : Nat
    a : α
    hd : G.IsRegularOfDegree d
    v : V
    ⊢ Eq ((SimpleGraph.adjMatrix α G).mulVec (Function.const V a) v) (HMul.hMul (↑ …
  -/
  simp [hd v]
  /-
    🎉 no goals
  -/


theorem adjMatrix_pow_apply_eq_card_walk [DecidableEq V] [Semiring α] (n : ℕ) (u v : V) :
    (G.adjMatrix α ^ n) u v = Fintype.card { p : G.Walk u v | p.length = n } := by
  /-
    V : Type u_1
    α : Type u_2
    G : SimpleGraph V
    inst✝³ : DecidableRel G.Adj
    inst✝² : Fintype V
    inst✝¹ : DecidableEq V
    inst✝ : Semiring α
    n : Nat
    u v : V
    ⊢ Eq (HPow.hPow (SimpleGraph.adjMatrix α G) n u v) ↑(Fintype.card ↑(setOf fun  …
  -/
  rw [card_set_walk_length_eq]
  /-
    V : Type u_1
    α : Type u_2
    G : SimpleGraph V
    inst✝³ : DecidableRel G.Adj
    inst✝² : Fintype V
    inst✝¹ : DecidableEq V
    inst✝ : Semiring α
    n : Nat
    u v : V
    ⊢ Eq (HPow.hPow (SimpleGraph.adjMatrix α G) n u v) ↑(G.finsetWalkLength n u v) …
  -/
  induction' n with n ih generalizing u v
    /-
      case zero
      V : Type u_1
      α : Type u_2
      G : SimpleGraph V
      inst✝³ : DecidableRel G.Adj
      inst✝² : Fintype V
      inst✝¹ : DecidableEq V
      inst✝ : Semiring α
      u v : V
      ⊢ Eq (HPow.hPow (SimpleGraph.adjMatrix α G) 0 u v) ↑(G.finsetWalkLength 0 u v) …
    -/
                                       /-
                                         🎉 no goals
                                       -/
  · obtain rfl | h := eq_or_ne u v <;> simp [finsetWalkLength, *]
                                       /-
                                         🎉 no goals
                                       -/
    /-
      case succ
      V : Type u_1
      α : Type u_2
      G : SimpleGraph V
      inst✝³ : DecidableRel G.Adj
      inst✝² : Fintype V
      inst✝¹ : DecidableEq V
      inst✝ : Semiring α
      n : Nat
      ih : ∀ (u v : V), Eq (HPow.hPow (SimpleGraph.adjMatrix α G) n u v) ↑(G.finsetW …
      u v : V
      ⊢ Eq (HPow.hPow (SimpleGraph.adjMatrix α G) (HAdd.hAdd n 1) u v) ↑(G.finsetWal …
    -/
  · simp only [pow_succ', finsetWalkLength, ih, adjMatrix_mul_apply]
    /-
      case succ
      V : Type u_1
      α : Type u_2
      G : SimpleGraph V
      inst✝³ : DecidableRel G.Adj
      inst✝² : Fintype V
      inst✝¹ : DecidableEq V
      inst✝ : Semiring α
      n : Nat
      ih : ∀ (u v : V), Eq (HPow.hPow (SimpleGraph.adjMatrix α G) n u v) ↑(G.finsetW …
      u v : V
      ⊢ Eq ((G.neighborFinset u).sum fun x => ↑(G.finsetWalkLength n x v).card) ↑(Fi …
    -/
    rw [Finset.card_biUnion]
      /-
        case succ
        V : Type u_1
        α : Type u_2
        G : SimpleGraph V
        inst✝³ : DecidableRel G.Adj
        inst✝² : Fintype V
        inst✝¹ : DecidableEq V
        inst✝ : Semiring α
        n : Nat
        ih : ∀ (u v : V), Eq (HPow.hPow (SimpleGraph.adjMatrix α G) n u v) ↑(G.finsetW …
        u v : V
        ⊢ Eq ((G.neighborFinset u).sum fun x => ↑(G.finsetWalkLength n x v).card) ↑(Fi …
      -/
    · norm_cast
      /-
        case succ
        V : Type u_1
        α : Type u_2
        G : SimpleGraph V
        inst✝³ : DecidableRel G.Adj
        inst✝² : Fintype V
        inst✝¹ : DecidableEq V
        inst✝ : Semiring α
        n : Nat
        ih : ∀ (u v : V), Eq (HPow.hPow (SimpleGraph.adjMatrix α G) n u v) ↑(G.finsetW …
        u v : V
        ⊢ Eq ↑((G.neighborFinset u).sum fun x => (G.finsetWalkLength n x v).card) ↑(Fi …
      -/
      simp only [Nat.cast_sum, card_map, neighborFinset_def]
      /-
        case succ
        V : Type u_1
        α : Type u_2
        G : SimpleGraph V
        inst✝³ : DecidableRel G.Adj
        inst✝² : Fintype V
        inst✝¹ : DecidableEq V
        inst✝ : Semiring α
        n : Nat
        ih : ∀ (u v : V), Eq (HPow.hPow (SimpleGraph.adjMatrix α G) n u v) ↑(G.finsetW …
        u v : V
        ⊢ Eq ((G.neighborSet u).toFinset.sum fun x => ↑(G.finsetWalkLength n x v).card …
      -/
      apply Finset.sum_toFinset_eq_subtype
      /-
        🎉 no goals
      -/
    -- Disjointness for card_bUnion
      /-
        case succ
        V : Type u_1
        α : Type u_2
        G : SimpleGraph V
        inst✝³ : DecidableRel G.Adj
        inst✝² : Fintype V
        inst✝¹ : DecidableEq V
        inst✝ : Semiring α
        n : Nat
        ih : ∀ (u v : V), Eq (HPow.hPow (SimpleGraph.adjMatrix α G) n u v) ↑(G.finsetW …
        u v : V
        ⊢ ∀ (x : ↑(G.neighborSet u)), Membership.mem Finset.univ x → ∀ (y : ↑(G.neighb …
      -/
    · rintro ⟨x, hx⟩ - ⟨y, hy⟩ - hxy
      /-
        case succ.mk.mk
        V : Type u_1
        α : Type u_2
        G : SimpleGraph V
        inst✝³ : DecidableRel G.Adj
        inst✝² : Fintype V
        inst✝¹ : DecidableEq V
        inst✝ : Semiring α
        n : Nat
        ih : ∀ (u v : V), Eq (HPow.hPow (SimpleGraph.adjMatrix α G) n u v) ↑(G.finsetW …
        u v x : V
        hx : Membership.mem (G.neighborSet u) x
        y : V
        hy : Membership.mem (G.neighborSet u) y
        hxy : Ne ⟨x, hx⟩ ⟨y, hy⟩
        ⊢ Disjoint (Finset.map { toFun := fun p => SimpleGraph.Walk.cons ⋯ p, inj' :=  …
      -/
      rw [disjoint_iff_inf_le]
      /-
        case succ.mk.mk
        V : Type u_1
        α : Type u_2
        G : SimpleGraph V
        inst✝³ : DecidableRel G.Adj
        inst✝² : Fintype V
        inst✝¹ : DecidableEq V
        inst✝ : Semiring α
        n : Nat
        ih : ∀ (u v : V), Eq (HPow.hPow (SimpleGraph.adjMatrix α G) n u v) ↑(G.finsetW …
        u v x : V
        hx : Membership.mem (G.neighborSet u) x
        y : V
        hy : Membership.mem (G.neighborSet u) y
        hxy : Ne ⟨x, hx⟩ ⟨y, hy⟩
        ⊢ LE.le (Min.min (Finset.map { toFun := fun p => SimpleGraph.Walk.cons ⋯ p, in …
      -/
      intro p hp
      /-
        case succ.mk.mk
        V : Type u_1
        α : Type u_2
        G : SimpleGraph V
        inst✝³ : DecidableRel G.Adj
        inst✝² : Fintype V
        inst✝¹ : DecidableEq V
        inst✝ : Semiring α
        n : Nat
        ih : ∀ (u v : V), Eq (HPow.hPow (SimpleGraph.adjMatrix α G) n u v) ↑(G.finsetW …
        u v x : V
        hx : Membership.mem (G.neighborSet u) x
        y : V
        hy : Membership.mem (G.neighborSet u) y
        hxy : Ne ⟨x, hx⟩ ⟨y, hy⟩
        p : G.Walk u v
        hp : Membership.mem (Min.min (Finset.map { toFun := fun p => SimpleGraph.Walk. …
        ⊢ Membership.mem Bot.bot p
      -/
      simp only [inf_eq_inter, mem_inter, mem_map, Function.Embedding.coeFn_mk, exists_prop] at hp
      /-
        case succ.mk.mk
        V : Type u_1
        α : Type u_2
        G : SimpleGraph V
        inst✝³ : DecidableRel G.Adj
        inst✝² : Fintype V
        inst✝¹ : DecidableEq V
        inst✝ : Semiring α
        n : Nat
        ih : ∀ (u v : V), Eq (HPow.hPow (SimpleGraph.adjMatrix α G) n u v) ↑(G.finsetW …
        u v x : V
        hx : Membership.mem (G.neighborSet u) x
        y : V
        hy : Membership.mem (G.neighborSet u) y
        hxy : Ne ⟨x, hx⟩ ⟨y, hy⟩
        p : G.Walk u v
        hp : And (Exists fun a => And (Membership.mem (G.finsetWalkLength n x v) a) (E …
        ⊢ Membership.mem Bot.bot p
      -/
      obtain ⟨⟨px, _, rfl⟩, ⟨py, hpy, hp⟩⟩ := hp
      /-
        case succ.mk.mk.intro.intro.intro.intro.intro
        V : Type u_1
        α : Type u_2
        G : SimpleGraph V
        inst✝³ : DecidableRel G.Adj
        inst✝² : Fintype V
        inst✝¹ : DecidableEq V
        inst✝ : Semiring α
        n : Nat
        ih : ∀ (u v : V), Eq (HPow.hPow (SimpleGraph.adjMatrix α G) n u v) ↑(G.finsetW …
        u v x : V
        hx : Membership.mem (G.neighborSet u) x
        y : V
        hy : Membership.mem (G.neighborSet u) y
        hxy : Ne ⟨x, hx⟩ ⟨y, hy⟩
        px : G.Walk x v
        left✝ : Membership.mem (G.finsetWalkLength n x v) px
        py : G.Walk y v
        hpy : Membership.mem (G.finsetWalkLength n y v) py
        hp : Eq (SimpleGraph.Walk.cons ⋯ py) (SimpleGraph.Walk.cons ⋯ px)
        ⊢ Membership.mem Bot.bot (SimpleGraph.Walk.cons ⋯ px)
      -/
      cases hp
      /-
        case succ.mk.mk.intro.intro.intro.intro.intro.refl
        V : Type u_1
        α : Type u_2
        G : SimpleGraph V
        inst✝³ : DecidableRel G.Adj
        inst✝² : Fintype V
        inst✝¹ : DecidableEq V
        inst✝ : Semiring α
        n : Nat
        ih : ∀ (u v : V), Eq (HPow.hPow (SimpleGraph.adjMatrix α G) n u v) ↑(G.finsetW …
        u v x : V
        hx : Membership.mem (G.neighborSet u) x
        px : G.Walk x v
        left✝ : Membership.mem (G.finsetWalkLength n x v) px
        hy : Membership.mem (G.neighborSet u) x
        hxy : Ne ⟨x, hx⟩ ⟨x, hy⟩
        hpy : Membership.mem (G.finsetWalkLength n x v) px
        ⊢ Membership.mem Bot.bot (SimpleGraph.Walk.cons ⋯ px)
      -/
      simp at hxy
      /-
        🎉 no goals
      -/


theorem dotProduct_mulVec_adjMatrix [NonAssocSemiring α] (x y : V → α) :
    x ⬝ᵥ (G.adjMatrix α).mulVec y = ∑ i : V, ∑ j : V, if G.Adj i j then x i * y j else 0 := by
  simp only [dotProduct, mulVec, adjMatrix_apply, ite_mul, one_mul, zero_mul, mul_sum, mul_ite,
    mul_zero]


/-- If `A` is qualified as an adjacency matrix,
    then the adjacency matrix of the graph induced by `A` is itself. -/
theorem adjMatrix_toGraph_eq [DecidableEq α] : h.toGraph.adjMatrix α = A := by
  /-
    V : Type u_1
    α : Type u_2
    inst✝² : MulZeroOneClass α
    inst✝¹ : Nontrivial α
    A : Matrix V V α
    h : A.IsAdjMatrix
    inst✝ : DecidableEq α
    ⊢ Eq (SimpleGraph.adjMatrix α h.toGraph) A
  -/
  ext i j
  /-
    case a
    V : Type u_1
    α : Type u_2
    inst✝² : MulZeroOneClass α
    inst✝¹ : Nontrivial α
    A : Matrix V V α
    h : A.IsAdjMatrix
    inst✝ : DecidableEq α
    i j : V
    ⊢ Eq (SimpleGraph.adjMatrix α h.toGraph i j) (A i j)
  -/
                                          /-
                                            🎉 no goals
                                          -/
  obtain h' | h' := h.zero_or_one i j <;> simp [h']
                                          /-
                                            🎉 no goals
                                          -/


