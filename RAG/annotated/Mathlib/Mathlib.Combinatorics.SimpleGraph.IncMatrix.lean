/-- `G.incMatrix R` is the `α × Sym2 α` matrix whose `(a, e)`-entry is `1` if `e` is incident to
`a` and `0` otherwise. -/
noncomputable def incMatrix [Zero R] [One R] : Matrix α (Sym2 α) R := fun a =>
  (G.incidenceSet a).indicator 1


theorem incMatrix_apply [Zero R] [One R] {a : α} {e : Sym2 α} :
    G.incMatrix R a e = (G.incidenceSet a).indicator 1 e :=
  rfl


/-- Entries of the incidence matrix can be computed given additional decidable instances. -/
theorem incMatrix_apply' [Zero R] [One R] [DecidableEq α] [DecidableRel G.Adj] {a : α}
    {e : Sym2 α} : G.incMatrix R a e = if e ∈ G.incidenceSet a then 1 else 0 := by
  /-
    R : Type u_1
    α : Type u_2
    G : SimpleGraph α
    inst✝³ : Zero R
    inst✝² : One R
    inst✝¹ : DecidableEq α
    inst✝ : DecidableRel G.Adj
    a : α
    e : Sym2 α
    ⊢ Eq (SimpleGraph.incMatrix R G a e) (ite (Membership.mem (G.incidenceSet a) e …
  -/
  unfold incMatrix Set.indicator
  /-
    R : Type u_1
    α : Type u_2
    G : SimpleGraph α
    inst✝³ : Zero R
    inst✝² : One R
    inst✝¹ : DecidableEq α
    inst✝ : DecidableRel G.Adj
    a : α
    e : Sym2 α
    ⊢ Eq (ite (Membership.mem (G.incidenceSet a) e) (1 e) 0) (ite (Membership.mem  …
  -/
  convert rfl
  /-
    🎉 no goals
  -/


theorem incMatrix_apply_mul_incMatrix_apply : G.incMatrix R a e * G.incMatrix R b e =
    (G.incidenceSet a ∩ G.incidenceSet b).indicator 1 e := by
  classical simp only [incMatrix, Set.indicator_apply, ite_zero_mul_ite_zero, Pi.one_apply, mul_one,
    Set.mem_inter_iff]


theorem incMatrix_apply_mul_incMatrix_apply_of_not_adj (hab : a ≠ b) (h : ¬G.Adj a b) :
    G.incMatrix R a e * G.incMatrix R b e = 0 := by
  /-
    R : Type u_1
    α : Type u_2
    G : SimpleGraph α
    inst✝ : MulZeroOneClass R
    a b : α
    e : Sym2 α
    hab : Ne a b
    h : Not (G.Adj a b)
    ⊢ Eq (HMul.hMul (SimpleGraph.incMatrix R G a e) (SimpleGraph.incMatrix R G b e …
  -/
  rw [incMatrix_apply_mul_incMatrix_apply, Set.indicator_of_not_mem]
  /-
    case h
    R : Type u_1
    α : Type u_2
    G : SimpleGraph α
    inst✝ : MulZeroOneClass R
    a b : α
    e : Sym2 α
    hab : Ne a b
    h : Not (G.Adj a b)
    ⊢ Not (Membership.mem (Inter.inter (G.incidenceSet a) (G.incidenceSet b)) e)
  -/
  rw [G.incidenceSet_inter_incidenceSet_of_not_adj h hab]
  /-
    case h
    R : Type u_1
    α : Type u_2
    G : SimpleGraph α
    inst✝ : MulZeroOneClass R
    a b : α
    e : Sym2 α
    hab : Ne a b
    h : Not (G.Adj a b)
    ⊢ Not (Membership.mem EmptyCollection.emptyCollection e)
  -/
  exact Set.not_mem_empty e
  /-
    🎉 no goals
  -/


theorem incMatrix_of_not_mem_incidenceSet (h : e ∉ G.incidenceSet a) : G.incMatrix R a e = 0 := by
  /-
    R : Type u_1
    α : Type u_2
    G : SimpleGraph α
    inst✝ : MulZeroOneClass R
    a : α
    e : Sym2 α
    h : Not (Membership.mem (G.incidenceSet a) e)
    ⊢ Eq (SimpleGraph.incMatrix R G a e) 0
  -/
  rw [incMatrix_apply, Set.indicator_of_not_mem h]
  /-
    🎉 no goals
  -/


theorem incMatrix_of_mem_incidenceSet (h : e ∈ G.incidenceSet a) : G.incMatrix R a e = 1 := by
  /-
    R : Type u_1
    α : Type u_2
    G : SimpleGraph α
    inst✝ : MulZeroOneClass R
    a : α
    e : Sym2 α
    h : Membership.mem (G.incidenceSet a) e
    ⊢ Eq (SimpleGraph.incMatrix R G a e) 1
  -/
  rw [incMatrix_apply, Set.indicator_of_mem h, Pi.one_apply]
  /-
    🎉 no goals
  -/


theorem incMatrix_apply_eq_zero_iff : G.incMatrix R a e = 0 ↔ e ∉ G.incidenceSet a := by
  /-
    R : Type u_1
    α : Type u_2
    G : SimpleGraph α
    inst✝¹ : MulZeroOneClass R
    a : α
    e : Sym2 α
    inst✝ : Nontrivial R
    ⊢ Iff (Eq (SimpleGraph.incMatrix R G a e) 0) (Not (Membership.mem (G.incidence …
  -/
  simp only [incMatrix_apply, Set.indicator_apply_eq_zero, Pi.one_apply, one_ne_zero]
  /-
    🎉 no goals
  -/


theorem incMatrix_apply_eq_one_iff : G.incMatrix R a e = 1 ↔ e ∈ G.incidenceSet a := by
  -- Porting note: was `convert one_ne_zero.ite_eq_left_iff; infer_instance`
  /-
    R : Type u_1
    α : Type u_2
    G : SimpleGraph α
    inst✝¹ : MulZeroOneClass R
    a : α
    e : Sym2 α
    inst✝ : Nontrivial R
    ⊢ Iff (Eq (SimpleGraph.incMatrix R G a e) 1) (Membership.mem (G.incidenceSet a …
  -/
  unfold incMatrix Set.indicator
  /-
    R : Type u_1
    α : Type u_2
    G : SimpleGraph α
    inst✝¹ : MulZeroOneClass R
    a : α
    e : Sym2 α
    inst✝ : Nontrivial R
    ⊢ Iff (Eq (ite (Membership.mem (G.incidenceSet a) e) (1 e) 0) 1) (Membership.m …
  -/
  simp only [Pi.one_apply]
  /-
    R : Type u_1
    α : Type u_2
    G : SimpleGraph α
    inst✝¹ : MulZeroOneClass R
    a : α
    e : Sym2 α
    inst✝ : Nontrivial R
    ⊢ Iff (Eq (ite (Membership.mem (G.incidenceSet a) e) 1 0) 1) (Membership.mem ( …
  -/
  apply Iff.intro <;> intro h
    /-
      case mp
      R : Type u_1
      α : Type u_2
      G : SimpleGraph α
      inst✝¹ : MulZeroOneClass R
      a : α
      e : Sym2 α
      inst✝ : Nontrivial R
      h : Eq (ite (Membership.mem (G.incidenceSet a) e) 1 0) 1
      ⊢ Membership.mem (G.incidenceSet a) e
    -/
                   /-
                     🎉 no goals
                   -/
  · split at h <;> simp_all only [zero_ne_one]
                   /-
                     🎉 no goals
                   -/
    /-
      case mpr
      R : Type u_1
      α : Type u_2
      G : SimpleGraph α
      inst✝¹ : MulZeroOneClass R
      a : α
      e : Sym2 α
      inst✝ : Nontrivial R
      h : Membership.mem (G.incidenceSet a) e
      ⊢ Eq (ite (Membership.mem (G.incidenceSet a) e) 1 0) 1
    -/
  · simp_all only [ite_true]
    /-
      🎉 no goals
    -/


theorem sum_incMatrix_apply [Fintype (Sym2 α)] [Fintype (neighborSet G a)] :
    ∑ e, G.incMatrix R a e = G.degree a := by
  /-
    R : Type u_1
    α : Type u_2
    G : SimpleGraph α
    inst✝² : NonAssocSemiring R
    a : α
    inst✝¹ : Fintype (Sym2 α)
    inst✝ : Fintype ↑(G.neighborSet a)
    ⊢ Eq (Finset.univ.sum fun e => SimpleGraph.incMatrix R G a e) ↑(G.degree a)
  -/
  classical simp [incMatrix_apply', sum_boole, Set.filter_mem_univ_eq_toFinset]
  /-
    🎉 no goals
  -/


theorem incMatrix_mul_transpose_diag [Fintype (Sym2 α)] [Fintype (neighborSet G a)] :
    (G.incMatrix R * (G.incMatrix R)ᵀ) a a = G.degree a := by
  classical
  rw [← sum_incMatrix_apply]
  simp only [mul_apply, incMatrix_apply', transpose_apply, mul_ite, mul_one, mul_zero]
  simp_all only [ite_true, sum_boole]


theorem sum_incMatrix_apply_of_mem_edgeSet [Fintype α] :
    e ∈ G.edgeSet → ∑ a, G.incMatrix R a e = 2 := by
  classical
    refine e.ind ?_
    intro a b h
    rw [mem_edgeSet] at h
    rw [← Nat.cast_two, ← card_pair h.ne]
    simp only [incMatrix_apply', sum_boole, mk'_mem_incidenceSet_iff, h]
    congr 2
    ext e
    simp only [mem_filter, mem_univ, true_and, mem_insert, mem_singleton]


theorem sum_incMatrix_apply_of_not_mem_edgeSet [Fintype α] (h : e ∉ G.edgeSet) :
    ∑ a, G.incMatrix R a e = 0 :=
  sum_eq_zero fun _ _ => G.incMatrix_of_not_mem_incidenceSet fun he => h he.1


theorem incMatrix_transpose_mul_diag [Fintype α] [Decidable (e ∈ G.edgeSet)] :
    ((G.incMatrix R)ᵀ * G.incMatrix R) e e = if e ∈ G.edgeSet then 2 else 0 := by
  classical
    simp only [Matrix.mul_apply, incMatrix_apply', transpose_apply, ite_zero_mul_ite_zero, one_mul,
      sum_boole, and_self_iff]
    split_ifs with h
    · revert h
      refine e.ind ?_
      intro v w h
      rw [← Nat.cast_two, ← card_pair (G.ne_of_adj h)]
      simp only [mk'_mem_incidenceSet_iff, G.mem_edgeSet.mp h, true_and, mem_univ, forall_true_left,
        forall_eq_or_imp, forall_eq, and_self, mem_singleton, ne_eq]
      congr 2
      ext u
      simp
    · revert h
      refine e.ind ?_
      intro v w h
      simp [mk'_mem_incidenceSet_iff, G.mem_edgeSet.not.mp h]


theorem incMatrix_mul_transpose_apply_of_adj (h : G.Adj a b) :
    (G.incMatrix R * (G.incMatrix R)ᵀ) a b = (1 : R) := by
  classical
    simp_rw [Matrix.mul_apply, Matrix.transpose_apply, incMatrix_apply_mul_incMatrix_apply,
      Set.indicator_apply, Pi.one_apply, sum_boole]
    convert @Nat.cast_one R _
    convert card_singleton s(a, b)
    rw [← coe_eq_singleton, coe_filter_univ]
    exact G.incidenceSet_inter_incidenceSet_of_adj h


theorem incMatrix_mul_transpose
    [∀ a, Fintype (neighborSet G a)] [DecidableEq α] [DecidableRel G.Adj] :
    G.incMatrix R * (G.incMatrix R)ᵀ = fun a b =>
      if a = b then (G.degree a : R) else if G.Adj a b then 1 else 0 := by
  /-
    R : Type u_1
    α : Type u_2
    G : SimpleGraph α
    inst✝⁴ : Fintype (Sym2 α)
    inst✝³ : Semiring R
    inst✝² : (a : α) → Fintype ↑(G.neighborSet a)
    inst✝¹ : DecidableEq α
    inst✝ : DecidableRel G.Adj
    ⊢ Eq (HMul.hMul (SimpleGraph.incMatrix R G) (SimpleGraph.incMatrix R G).transp …
  -/
  ext a b
  /-
    case a
    R : Type u_1
    α : Type u_2
    G : SimpleGraph α
    inst✝⁴ : Fintype (Sym2 α)
    inst✝³ : Semiring R
    inst✝² : (a : α) → Fintype ↑(G.neighborSet a)
    inst✝¹ : DecidableEq α
    inst✝ : DecidableRel G.Adj
    a b : α
    ⊢ Eq (HMul.hMul (SimpleGraph.incMatrix R G) (SimpleGraph.incMatrix R G).transp …
  -/
  split_ifs with h h'
    /-
      case pos
      R : Type u_1
      α : Type u_2
      G : SimpleGraph α
      inst✝⁴ : Fintype (Sym2 α)
      inst✝³ : Semiring R
      inst✝² : (a : α) → Fintype ↑(G.neighborSet a)
      inst✝¹ : DecidableEq α
      inst✝ : DecidableRel G.Adj
      a b : α
      h : Eq a b
      ⊢ Eq (HMul.hMul (SimpleGraph.incMatrix R G) (SimpleGraph.incMatrix R G).transp …
    -/
  · subst b
    /-
      case pos
      R : Type u_1
      α : Type u_2
      G : SimpleGraph α
      inst✝⁴ : Fintype (Sym2 α)
      inst✝³ : Semiring R
      inst✝² : (a : α) → Fintype ↑(G.neighborSet a)
      inst✝¹ : DecidableEq α
      inst✝ : DecidableRel G.Adj
      a : α
      ⊢ Eq (HMul.hMul (SimpleGraph.incMatrix R G) (SimpleGraph.incMatrix R G).transp …
    -/
    exact incMatrix_mul_transpose_diag (R := R) G
    /-
      🎉 no goals
    -/
    /-
      case pos
      R : Type u_1
      α : Type u_2
      G : SimpleGraph α
      inst✝⁴ : Fintype (Sym2 α)
      inst✝³ : Semiring R
      inst✝² : (a : α) → Fintype ↑(G.neighborSet a)
      inst✝¹ : DecidableEq α
      inst✝ : DecidableRel G.Adj
      a b : α
      h : Not (Eq a b)
      h' : G.Adj a b
      ⊢ Eq (HMul.hMul (SimpleGraph.incMatrix R G) (SimpleGraph.incMatrix R G).transp …
    -/
  · exact G.incMatrix_mul_transpose_apply_of_adj h'
    /-
      🎉 no goals
    -/
  · simp only [Matrix.mul_apply, Matrix.transpose_apply,
      G.incMatrix_apply_mul_incMatrix_apply_of_not_adj h h', sum_const_zero]


