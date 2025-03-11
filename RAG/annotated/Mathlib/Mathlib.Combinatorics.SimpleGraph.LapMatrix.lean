theorem degree_eq_sum_if_adj {R : Type*} [AddCommMonoidWithOne R] (i : V) :
    (G.degree i : R) = ∑ j : V, if G.Adj i j then 1 else 0 := by
  /-
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    R : Type u_3
    inst✝ : AddCommMonoidWithOne R
    i : V
    ⊢ Eq (↑(G.degree i)) (Finset.univ.sum fun j => ite (G.Adj i j) 1 0)
  -/
  unfold degree neighborFinset neighborSet
  /-
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    R : Type u_3
    inst✝ : AddCommMonoidWithOne R
    i : V
    ⊢ Eq (↑(setOf fun w => G.Adj i w).toFinset.card) (Finset.univ.sum fun j => ite …
  -/
  rw [sum_boole, Set.toFinset_setOf]
  /-
    🎉 no goals
  -/


/-- The diagonal matrix consisting of the degrees of the vertices in the graph. -/
def degMatrix [AddMonoidWithOne R] : Matrix V V R := Matrix.diagonal (G.degree ·)


/-- The *Laplacian matrix* `lapMatrix G R` of a graph `G`
is the matrix `L = D - A` where `D` is the degree and `A` the adjacency matrix of `G`. -/
def lapMatrix [AddGroupWithOne R] : Matrix V V R := G.degMatrix R - G.adjMatrix R


theorem isSymm_degMatrix [AddMonoidWithOne R] : (G.degMatrix R).IsSymm :=
  isSymm_diagonal _


theorem isSymm_lapMatrix [AddGroupWithOne R] : (G.lapMatrix R).IsSymm :=
  (isSymm_degMatrix _).sub (isSymm_adjMatrix _)


theorem degMatrix_mulVec_apply [NonAssocSemiring R] (v : V) (vec : V → R) :
    (G.degMatrix R *ᵥ vec) v = G.degree v * vec v := by
  /-
    V : Type u_1
    R : Type u_2
    inst✝³ : Fintype V
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : DecidableEq V
    inst✝ : NonAssocSemiring R
    v : V
    vec : V → R
    ⊢ Eq ((SimpleGraph.degMatrix R G).mulVec vec v) (HMul.hMul (↑(G.degree v)) (ve …
  -/
  rw [degMatrix, mulVec_diagonal]
  /-
    🎉 no goals
  -/


theorem lapMatrix_mulVec_apply [NonAssocRing R] (v : V) (vec : V → R) :
    (G.lapMatrix R *ᵥ vec) v = G.degree v * vec v - ∑ u ∈ G.neighborFinset v, vec u := by
  /-
    V : Type u_1
    R : Type u_2
    inst✝³ : Fintype V
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : DecidableEq V
    inst✝ : NonAssocRing R
    v : V
    vec : V → R
    ⊢ Eq ((SimpleGraph.lapMatrix R G).mulVec vec v) (HSub.hSub (HMul.hMul (↑(G.deg …
  -/
  simp_rw [lapMatrix, sub_mulVec, Pi.sub_apply, degMatrix_mulVec_apply, adjMatrix_mulVec_apply]
  /-
    🎉 no goals
  -/


theorem lapMatrix_mulVec_const_eq_zero [Ring R] : mulVec (G.lapMatrix R) (fun _ ↦ 1) = 0 := by
  /-
    V : Type u_1
    R : Type u_2
    inst✝³ : Fintype V
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : DecidableEq V
    inst✝ : Ring R
    ⊢ Eq ((SimpleGraph.lapMatrix R G).mulVec fun x => 1) 0
  -/
  ext1 i
  /-
    case h
    V : Type u_1
    R : Type u_2
    inst✝³ : Fintype V
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : DecidableEq V
    inst✝ : Ring R
    i : V
    ⊢ Eq ((SimpleGraph.lapMatrix R G).mulVec (fun x => 1) i) (0 i)
  -/
  rw [lapMatrix_mulVec_apply]
  /-
    case h
    V : Type u_1
    R : Type u_2
    inst✝³ : Fintype V
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : DecidableEq V
    inst✝ : Ring R
    i : V
    ⊢ Eq (HSub.hSub (HMul.hMul (↑(G.degree i)) 1) ((G.neighborFinset i).sum fun u  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem dotProduct_mulVec_degMatrix [CommRing R] (x : V → R) :
    x ⬝ᵥ (G.degMatrix R *ᵥ x) = ∑ i : V, G.degree i * x i * x i := by
  /-
    V : Type u_1
    R : Type u_2
    inst✝³ : Fintype V
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : DecidableEq V
    inst✝ : CommRing R
    x : V → R
    ⊢ Eq (dotProduct x ((SimpleGraph.degMatrix R G).mulVec x)) (Finset.univ.sum fu …
  -/
  simp only [dotProduct, degMatrix, mulVec_diagonal, ← mul_assoc, mul_comm]
  /-
    🎉 no goals
  -/


/-- Let $L$ be the graph Laplacian and let $x \in \mathbb{R}$, then
$$x^{\top} L x = \sum_{i \sim j} (x_{i}-x_{j})^{2}$$,
where $\sim$ denotes the adjacency relation -/
theorem lapMatrix_toLinearMap₂' [Field R] [CharZero R] (x : V → R) :
    toLinearMap₂' R (G.lapMatrix R) x x =
    (∑ i : V, ∑ j : V, if G.Adj i j then (x i - x j)^2 else 0) / 2 := by
  simp_rw [toLinearMap₂'_apply', lapMatrix, sub_mulVec, dotProduct_sub, dotProduct_mulVec_degMatrix,
    dotProduct_mulVec_adjMatrix, ← sum_sub_distrib, degree_eq_sum_if_adj, sum_mul, ite_mul, one_mul,
    zero_mul, ← sum_sub_distrib, ite_sub_ite, sub_zero]
  /-
    V : Type u_1
    R : Type u_2
    inst✝⁴ : Fintype V
    G : SimpleGraph V
    inst✝³ : DecidableRel G.Adj
    inst✝² : DecidableEq V
    inst✝¹ : Field R
    inst✝ : CharZero R
    x : V → R
    ⊢ Eq (Finset.univ.sum fun x_1 => Finset.univ.sum fun x_2 => ite (G.Adj x_1 x_2 …
  -/
  rw [← add_self_div_two (∑ x_1 : V, ∑ x_2 : V, _)]
  /-
    V : Type u_1
    R : Type u_2
    inst✝⁴ : Fintype V
    G : SimpleGraph V
    inst✝³ : DecidableRel G.Adj
    inst✝² : DecidableEq V
    inst✝¹ : Field R
    inst✝ : CharZero R
    x : V → R
    ⊢ Eq (HDiv.hDiv (HAdd.hAdd (Finset.univ.sum fun x_1 => Finset.univ.sum fun x_2 …
  -/
  conv_lhs => enter [1,2,2,i,2,j]; rw [if_congr (adj_comm G i j) rfl rfl]
  /-
    V : Type u_1
    R : Type u_2
    inst✝⁴ : Fintype V
    G : SimpleGraph V
    inst✝³ : DecidableRel G.Adj
    inst✝² : DecidableEq V
    inst✝¹ : Field R
    inst✝ : CharZero R
    x : V → R
    ⊢ Eq (HDiv.hDiv (HAdd.hAdd (Finset.univ.sum fun x_1 => Finset.univ.sum fun x_2 …
  -/
  conv_lhs => enter [1,2]; rw [Finset.sum_comm]
  /-
    V : Type u_1
    R : Type u_2
    inst✝⁴ : Fintype V
    G : SimpleGraph V
    inst✝³ : DecidableRel G.Adj
    inst✝² : DecidableEq V
    inst✝¹ : Field R
    inst✝ : CharZero R
    x : V → R
    ⊢ Eq (HDiv.hDiv (HAdd.hAdd (Finset.univ.sum fun x_1 => Finset.univ.sum fun x_2 …
  -/
  simp_rw [← sum_add_distrib, ite_add_ite]
  /-
    V : Type u_1
    R : Type u_2
    inst✝⁴ : Fintype V
    G : SimpleGraph V
    inst✝³ : DecidableRel G.Adj
    inst✝² : DecidableEq V
    inst✝¹ : Field R
    inst✝ : CharZero R
    x : V → R
    ⊢ Eq (HDiv.hDiv (Finset.univ.sum fun x_1 => Finset.univ.sum fun x_2 => ite (G. …
  -/
  congr 2 with i
  /-
    case e_a.e_f.h
    V : Type u_1
    R : Type u_2
    inst✝⁴ : Fintype V
    G : SimpleGraph V
    inst✝³ : DecidableRel G.Adj
    inst✝² : DecidableEq V
    inst✝¹ : Field R
    inst✝ : CharZero R
    x : V → R
    i : V
    ⊢ Eq (Finset.univ.sum fun x_1 => ite (G.Adj i x_1) (HAdd.hAdd (HSub.hSub (HMul …
  -/
  congr 2 with j
  /-
    case e_a.e_f.h.e_f.h
    V : Type u_1
    R : Type u_2
    inst✝⁴ : Fintype V
    G : SimpleGraph V
    inst✝³ : DecidableRel G.Adj
    inst✝² : DecidableEq V
    inst✝¹ : Field R
    inst✝ : CharZero R
    x : V → R
    i j : V
    ⊢ Eq (ite (G.Adj i j) (HAdd.hAdd (HSub.hSub (HMul.hMul (x i) (x i)) (HMul.hMul …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


/-- The Laplacian matrix is positive semidefinite -/
theorem posSemidef_lapMatrix [LinearOrderedField R] [StarRing R]
    [TrivialStar R] : PosSemidef (G.lapMatrix R) := by
  /-
    V : Type u_1
    R : Type u_2
    inst✝⁵ : Fintype V
    G : SimpleGraph V
    inst✝⁴ : DecidableRel G.Adj
    inst✝³ : DecidableEq V
    inst✝² : LinearOrderedField R
    inst✝¹ : StarRing R
    inst✝ : TrivialStar R
    ⊢ (SimpleGraph.lapMatrix R G).PosSemidef
  -/
  constructor
    /-
      case left
      V : Type u_1
      R : Type u_2
      inst✝⁵ : Fintype V
      G : SimpleGraph V
      inst✝⁴ : DecidableRel G.Adj
      inst✝³ : DecidableEq V
      inst✝² : LinearOrderedField R
      inst✝¹ : StarRing R
      inst✝ : TrivialStar R
      ⊢ (SimpleGraph.lapMatrix R G).IsHermitian
    -/
  · rw [IsHermitian, conjTranspose_eq_transpose_of_trivial, isSymm_lapMatrix]
    /-
      🎉 no goals
    -/
    /-
      case right
      V : Type u_1
      R : Type u_2
      inst✝⁵ : Fintype V
      G : SimpleGraph V
      inst✝⁴ : DecidableRel G.Adj
      inst✝³ : DecidableEq V
      inst✝² : LinearOrderedField R
      inst✝¹ : StarRing R
      inst✝ : TrivialStar R
      ⊢ ∀ (x : V → R), LE.le 0 (dotProduct (Star.star x) ((SimpleGraph.lapMatrix R G …
    -/
  · intro x
    /-
      case right
      V : Type u_1
      R : Type u_2
      inst✝⁵ : Fintype V
      G : SimpleGraph V
      inst✝⁴ : DecidableRel G.Adj
      inst✝³ : DecidableEq V
      inst✝² : LinearOrderedField R
      inst✝¹ : StarRing R
      inst✝ : TrivialStar R
      x : V → R
      ⊢ LE.le 0 (dotProduct (Star.star x) ((SimpleGraph.lapMatrix R G).mulVec x))
    -/
    rw [star_trivial, ← toLinearMap₂'_apply', lapMatrix_toLinearMap₂']
    /-
      case right
      V : Type u_1
      R : Type u_2
      inst✝⁵ : Fintype V
      G : SimpleGraph V
      inst✝⁴ : DecidableRel G.Adj
      inst✝³ : DecidableEq V
      inst✝² : LinearOrderedField R
      inst✝¹ : StarRing R
      inst✝ : TrivialStar R
      x : V → R
      ⊢ LE.le 0 (HDiv.hDiv (Finset.univ.sum fun i => Finset.univ.sum fun j => ite (G …
    -/
    positivity
    /-
      🎉 no goals
    -/


theorem lapMatrix_toLinearMap₂'_apply'_eq_zero_iff_forall_adj [LinearOrderedField R] (x : V → R) :
    Matrix.toLinearMap₂' R (G.lapMatrix R) x x = 0 ↔ ∀ i j : V, G.Adj i j → x i = x j := by
  simp (disch := intros; positivity)
    [lapMatrix_toLinearMap₂', sum_eq_zero_iff_of_nonneg, sub_eq_zero]


theorem lapMatrix_toLin'_apply_eq_zero_iff_forall_adj (x : V → ℝ) :
    Matrix.toLin' (G.lapMatrix ℝ) x = 0 ↔ ∀ i j : V, G.Adj i j → x i = x j := by
  rw [← (posSemidef_lapMatrix ℝ G).toLinearMap₂'_zero_iff, star_trivial,
      lapMatrix_toLinearMap₂'_apply'_eq_zero_iff_forall_adj]


theorem lapMatrix_toLinearMap₂'_apply'_eq_zero_iff_forall_reachable (x : V → ℝ) :
    Matrix.toLinearMap₂' ℝ (G.lapMatrix ℝ) x x = 0 ↔
      ∀ i j : V, G.Reachable i j → x i = x j := by
  /-
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : DecidableEq V
    x : V → Real
    ⊢ Iff (Eq ((((Matrix.toLinearMap₂' Real) (SimpleGraph.lapMatrix Real G)) x) x) …
  -/
  rw [lapMatrix_toLinearMap₂'_apply'_eq_zero_iff_forall_adj]
  /-
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : DecidableEq V
    x : V → Real
    ⊢ Iff (∀ (i j : V), G.Adj i j → Eq (x i) (x j)) (∀ (i j : V), G.Reachable i j  …
  -/
  refine ⟨?_, fun h i j hA ↦ h i j hA.reachable⟩
  /-
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : DecidableEq V
    x : V → Real
    ⊢ (∀ (i j : V), G.Adj i j → Eq (x i) (x j)) → ∀ (i j : V), G.Reachable i j → E …
  -/
  intro h i j ⟨w⟩
  induction w with
  | nil => rfl
  | cons hA _ h' => exact (h _ _ hA).trans h'


theorem lapMatrix_toLin'_apply_eq_zero_iff_forall_reachable (x : V → ℝ) :
    Matrix.toLin' (G.lapMatrix ℝ) x = 0 ↔ ∀ i j : V, G.Reachable i j → x i = x j := by
  rw [← (posSemidef_lapMatrix ℝ G).toLinearMap₂'_zero_iff, star_trivial,
      lapMatrix_toLinearMap₂'_apply'_eq_zero_iff_forall_reachable]


lemma mem_ker_toLin'_lapMatrix_of_connectedComponent {G : SimpleGraph V} [DecidableRel G.Adj]
    [DecidableEq G.ConnectedComponent] (c : G.ConnectedComponent) :
    (fun i ↦ if connectedComponentMk G i = c then 1 else 0) ∈
      LinearMap.ker (toLin' (lapMatrix ℝ G)) := by
  /-
    V : Type u_1
    inst✝³ : Fintype V
    inst✝² : DecidableEq V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : DecidableEq G.ConnectedComponent
    c : G.ConnectedComponent
    ⊢ Membership.mem (LinearMap.ker (Matrix.toLin' (SimpleGraph.lapMatrix Real G)) …
  -/
  rw [LinearMap.mem_ker, lapMatrix_toLin'_apply_eq_zero_iff_forall_reachable]
  /-
    V : Type u_1
    inst✝³ : Fintype V
    inst✝² : DecidableEq V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : DecidableEq G.ConnectedComponent
    c : G.ConnectedComponent
    ⊢ ∀ (i j : V), G.Reachable i j → Eq (ite (Eq (G.connectedComponentMk i) c) 1 0 …
  -/
  intro i j h
  /-
    V : Type u_1
    inst✝³ : Fintype V
    inst✝² : DecidableEq V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : DecidableEq G.ConnectedComponent
    c : G.ConnectedComponent
    i j : V
    h : G.Reachable i j
    ⊢ Eq (ite (Eq (G.connectedComponentMk i) c) 1 0) (ite (Eq (G.connectedComponen …
  -/
  split_ifs with h₁ h₂ h₃
    /-
      case pos
      V : Type u_1
      inst✝³ : Fintype V
      inst✝² : DecidableEq V
      G : SimpleGraph V
      inst✝¹ : DecidableRel G.Adj
      inst✝ : DecidableEq G.ConnectedComponent
      c : G.ConnectedComponent
      i j : V
      h : G.Reachable i j
      h₁ : Eq (G.connectedComponentMk i) c
      h₂ : Eq (G.connectedComponentMk j) c
      ⊢ Eq 1 1
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      V : Type u_1
      inst✝³ : Fintype V
      inst✝² : DecidableEq V
      G : SimpleGraph V
      inst✝¹ : DecidableRel G.Adj
      inst✝ : DecidableEq G.ConnectedComponent
      c : G.ConnectedComponent
      i j : V
      h : G.Reachable i j
      h₁ : Eq (G.connectedComponentMk i) c
      h₂ : Not (Eq (G.connectedComponentMk j) c)
      ⊢ Eq 1 0
    -/
  · rw [← ConnectedComponent.eq] at h
    /-
      case neg
      V : Type u_1
      inst✝³ : Fintype V
      inst✝² : DecidableEq V
      G : SimpleGraph V
      inst✝¹ : DecidableRel G.Adj
      inst✝ : DecidableEq G.ConnectedComponent
      c : G.ConnectedComponent
      i j : V
      h : Eq (G.connectedComponentMk i) (G.connectedComponentMk j)
      h₁ : Eq (G.connectedComponentMk i) c
      h₂ : Not (Eq (G.connectedComponentMk j) c)
      ⊢ Eq 1 0
    -/
    exact (h₂ (h₁ ▸ h.symm)).elim
    /-
      🎉 no goals
    -/
    /-
      case pos
      V : Type u_1
      inst✝³ : Fintype V
      inst✝² : DecidableEq V
      G : SimpleGraph V
      inst✝¹ : DecidableRel G.Adj
      inst✝ : DecidableEq G.ConnectedComponent
      c : G.ConnectedComponent
      i j : V
      h : G.Reachable i j
      h₁ : Not (Eq (G.connectedComponentMk i) c)
      h₃ : Eq (G.connectedComponentMk j) c
      ⊢ Eq 0 1
    -/
  · rw [← ConnectedComponent.eq] at h
    /-
      case pos
      V : Type u_1
      inst✝³ : Fintype V
      inst✝² : DecidableEq V
      G : SimpleGraph V
      inst✝¹ : DecidableRel G.Adj
      inst✝ : DecidableEq G.ConnectedComponent
      c : G.ConnectedComponent
      i j : V
      h : Eq (G.connectedComponentMk i) (G.connectedComponentMk j)
      h₁ : Not (Eq (G.connectedComponentMk i) c)
      h₃ : Eq (G.connectedComponentMk j) c
      ⊢ Eq 0 1
    -/
    exact (h₁ (h₃ ▸ h)).elim
    /-
      🎉 no goals
    -/
    /-
      case neg
      V : Type u_1
      inst✝³ : Fintype V
      inst✝² : DecidableEq V
      G : SimpleGraph V
      inst✝¹ : DecidableRel G.Adj
      inst✝ : DecidableEq G.ConnectedComponent
      c : G.ConnectedComponent
      i j : V
      h : G.Reachable i j
      h₁ : Not (Eq (G.connectedComponentMk i) c)
      h₃ : Not (Eq (G.connectedComponentMk j) c)
      ⊢ Eq 0 0
    -/
  · rfl
    /-
      🎉 no goals
    -/


/-- Given a connected component `c` of a graph `G`, `lapMatrix_ker_basis_aux c` is the map
`V → ℝ` which is `1` on the vertices in `c` and `0` elsewhere.
The family of these maps indexed by the connected components of `G` proves to be a basis
of the kernel of `lapMatrix G R` -/
def lapMatrix_ker_basis_aux (c : G.ConnectedComponent) :
    LinearMap.ker (Matrix.toLin' (G.lapMatrix ℝ)) :=
  ⟨fun i ↦ if G.connectedComponentMk i = c then (1 : ℝ)  else 0,
    mem_ker_toLin'_lapMatrix_of_connectedComponent c⟩


lemma linearIndependent_lapMatrix_ker_basis_aux :
    LinearIndependent ℝ (lapMatrix_ker_basis_aux G) := by
  /-
    V : Type u_1
    inst✝³ : Fintype V
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : DecidableEq V
    inst✝ : DecidableEq G.ConnectedComponent
    ⊢ LinearIndependent Real G.lapMatrix_ker_basis_aux
  -/
  rw [Fintype.linearIndependent_iff]
  /-
    V : Type u_1
    inst✝³ : Fintype V
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : DecidableEq V
    inst✝ : DecidableEq G.ConnectedComponent
    ⊢ ∀ (g : G.ConnectedComponent → Real), Eq (Finset.univ.sum fun i => HSMul.hSMu …
  -/
  intro g h0
  /-
    V : Type u_1
    inst✝³ : Fintype V
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : DecidableEq V
    inst✝ : DecidableEq G.ConnectedComponent
    g : G.ConnectedComponent → Real
    h0 : Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) (G.lapMatrix_ker_basis_aux …
    ⊢ ∀ (i : G.ConnectedComponent), Eq (g i) 0
  -/
  rw [Subtype.ext_iff] at h0
  have h : ∑ c, g c • lapMatrix_ker_basis_aux G c = fun i ↦ g (connectedComponentMk G i) := by
    simp only [lapMatrix_ker_basis_aux, SetLike.mk_smul_mk, AddSubmonoid.coe_finset_sum]
    repeat rw [AddSubmonoid.coe_finset_sum]
    ext i
    simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, mul_ite, mul_one, mul_zero, sum_ite_eq,
      mem_univ, ↓reduceIte]
  /-
    V : Type u_1
    inst✝³ : Fintype V
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : DecidableEq V
    inst✝ : DecidableEq G.ConnectedComponent
    g : G.ConnectedComponent → Real
    h0 : Eq ↑(Finset.univ.sum fun i => HSMul.hSMul (g i) (G.lapMatrix_ker_basis_au …
    h : Eq ↑(Finset.univ.sum fun c => HSMul.hSMul (g c) (G.lapMatrix_ker_basis_aux …
    ⊢ ∀ (i : G.ConnectedComponent), Eq (g i) 0
  -/
  rw [h] at h0
  /-
    V : Type u_1
    inst✝³ : Fintype V
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : DecidableEq V
    inst✝ : DecidableEq G.ConnectedComponent
    g : G.ConnectedComponent → Real
    h0 : Eq (fun i => g (G.connectedComponentMk i)) ↑0
    h : Eq ↑(Finset.univ.sum fun c => HSMul.hSMul (g c) (G.lapMatrix_ker_basis_aux …
    ⊢ ∀ (i : G.ConnectedComponent), Eq (g i) 0
  -/
  intro c
  /-
    V : Type u_1
    inst✝³ : Fintype V
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : DecidableEq V
    inst✝ : DecidableEq G.ConnectedComponent
    g : G.ConnectedComponent → Real
    h0 : Eq (fun i => g (G.connectedComponentMk i)) ↑0
    h : Eq ↑(Finset.univ.sum fun c => HSMul.hSMul (g c) (G.lapMatrix_ker_basis_aux …
    c : G.ConnectedComponent
    ⊢ Eq (g c) 0
  -/
  obtain ⟨i, h'⟩ : ∃ i : V, G.connectedComponentMk i = c := Quot.exists_rep c
  /-
    case intro
    V : Type u_1
    inst✝³ : Fintype V
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : DecidableEq V
    inst✝ : DecidableEq G.ConnectedComponent
    g : G.ConnectedComponent → Real
    h0 : Eq (fun i => g (G.connectedComponentMk i)) ↑0
    h : Eq ↑(Finset.univ.sum fun c => HSMul.hSMul (g c) (G.lapMatrix_ker_basis_aux …
    c : G.ConnectedComponent
    i : V
    h' : Eq (G.connectedComponentMk i) c
    ⊢ Eq (g c) 0
  -/
  exact h' ▸ congrFun h0 i
  /-
    🎉 no goals
  -/


lemma top_le_span_range_lapMatrix_ker_basis_aux :
    ⊤ ≤ Submodule.span ℝ (Set.range (lapMatrix_ker_basis_aux G)) := by
  /-
    V : Type u_1
    inst✝³ : Fintype V
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : DecidableEq V
    inst✝ : DecidableEq G.ConnectedComponent
    ⊢ LE.le Top.top (Submodule.span Real (Set.range G.lapMatrix_ker_basis_aux))
  -/
  intro x _
  /-
    V : Type u_1
    inst✝³ : Fintype V
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : DecidableEq V
    inst✝ : DecidableEq G.ConnectedComponent
    x : Subtype fun x => Membership.mem (LinearMap.ker (Matrix.toLin' (SimpleGraph …
    a✝ : Membership.mem Top.top x
    ⊢ Membership.mem (Submodule.span Real (Set.range G.lapMatrix_ker_basis_aux)) x
  -/
  rw [mem_span_range_iff_exists_fun]
  use Quot.lift x.val (by rw [← lapMatrix_toLin'_apply_eq_zero_iff_forall_reachable G x,
    LinearMap.map_coe_ker])
  /-
    case h
    V : Type u_1
    inst✝³ : Fintype V
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : DecidableEq V
    inst✝ : DecidableEq G.ConnectedComponent
    x : Subtype fun x => Membership.mem (LinearMap.ker (Matrix.toLin' (SimpleGraph …
    a✝ : Membership.mem Top.top x
    ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul (Quot.lift ↑x ⋯ i) (G.lapMatrix_ker …
  -/
  ext j
  /-
    case h.a.h
    V : Type u_1
    inst✝³ : Fintype V
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : DecidableEq V
    inst✝ : DecidableEq G.ConnectedComponent
    x : Subtype fun x => Membership.mem (LinearMap.ker (Matrix.toLin' (SimpleGraph …
    a✝ : Membership.mem Top.top x
    j : V
    ⊢ Eq (↑(Finset.univ.sum fun i => HSMul.hSMul (Quot.lift ↑x ⋯ i) (G.lapMatrix_k …
  -/
  simp only [lapMatrix_ker_basis_aux]
  /-
    case h.a.h
    V : Type u_1
    inst✝³ : Fintype V
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : DecidableEq V
    inst✝ : DecidableEq G.ConnectedComponent
    x : Subtype fun x => Membership.mem (LinearMap.ker (Matrix.toLin' (SimpleGraph …
    a✝ : Membership.mem Top.top x
    j : V
    ⊢ Eq (↑(Finset.univ.sum fun x_1 => HSMul.hSMul (Quot.lift ↑x ⋯ x_1) ⟨fun i =>  …
  -/
  rw [AddSubmonoid.coe_finset_sum]
  simp only [SetLike.mk_smul_mk, Finset.sum_apply, Pi.smul_apply, smul_eq_mul, mul_ite, mul_one,
    mul_zero, sum_ite_eq, mem_univ, ↓reduceIte]
  /-
    case h.a.h
    V : Type u_1
    inst✝³ : Fintype V
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    inst✝¹ : DecidableEq V
    inst✝ : DecidableEq G.ConnectedComponent
    x : Subtype fun x => Membership.mem (LinearMap.ker (Matrix.toLin' (SimpleGraph …
    a✝ : Membership.mem Top.top x
    j : V
    ⊢ Eq (Quot.lift ↑x ⋯ (G.connectedComponentMk j)) (↑x j)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `lapMatrix_ker_basis G` is a basis of the nullspace indexed by its connected components,
the basis is made up of the functions `V → ℝ` which are `1` on the vertices of the given
connected component and `0` elsewhere. -/
noncomputable def lapMatrix_ker_basis :=
  Basis.mk (linearIndependent_lapMatrix_ker_basis_aux G)
    (top_le_span_range_lapMatrix_ker_basis_aux G)


/-- The number of connected components in `G` is the dimension of the nullspace of its Laplacian. -/
theorem card_ConnectedComponent_eq_rank_ker_lapMatrix : Fintype.card G.ConnectedComponent =
    Module.finrank ℝ (LinearMap.ker (Matrix.toLin' (G.lapMatrix ℝ))) := by
  classical
  rw [Module.finrank_eq_card_basis (lapMatrix_ker_basis G)]


