/-- From a `β`-scheme on `α` `A`, we define a partial function from `(ℕ → β)` to `α`
which sends each infinite sequence `x` to an element of the intersection along the
branch corresponding to `x`, if it exists.
We call this the map induced by the scheme. -/
noncomputable def inducedMap : Σs : Set (ℕ → β), s → α :=
  ⟨fun x => Set.Nonempty (⋂ n : ℕ, A (res x n)), fun x => x.property.some⟩


/-- A scheme is antitone if each set contains its children. -/
protected def Antitone : Prop :=
  ∀ l : List β, ∀ a : β, A (a :: l) ⊆ A l


/-- A useful strengthening of being antitone is to require that each set contains
the closure of each of its children. -/
def ClosureAntitone [TopologicalSpace α] : Prop :=
  ∀ l : List β, ∀ a : β, closure (A (a :: l)) ⊆ A l


/-- A scheme is disjoint if the children of each set of pairwise disjoint. -/
protected def Disjoint : Prop :=
  ∀ l : List β, Pairwise fun a b => Disjoint (A (a :: l)) (A (b :: l))


/-- If `x` is in the domain of the induced map of a scheme `A`,
its image under this map is in each set along the corresponding branch. -/
theorem map_mem (x : (inducedMap A).1) (n : ℕ) : (inducedMap A).2 x ∈ A (res x n) := by
  /-
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    x : ↑(CantorScheme.inducedMap A).fst
    n : Nat
    ⊢ Membership.mem (A (PiNat.res (↑x) n)) ((CantorScheme.inducedMap A).snd x)
  -/
  have := x.property.some_mem
  /-
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    x : ↑(CantorScheme.inducedMap A).fst
    n : Nat
    this : Membership.mem (Set.iInter fun n => A (PiNat.res (↑x) n)) (Set.Nonempty …
    ⊢ Membership.mem (A (PiNat.res (↑x) n)) ((CantorScheme.inducedMap A).snd x)
  -/
  rw [mem_iInter] at this
  /-
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    x : ↑(CantorScheme.inducedMap A).fst
    n : Nat
    this : ∀ (i : Nat), Membership.mem (A (PiNat.res (↑x) i)) (Set.Nonempty.some ⋯)
    ⊢ Membership.mem (A (PiNat.res (↑x) n)) ((CantorScheme.inducedMap A).snd x)
  -/
  exact this n
  /-
    🎉 no goals
  -/


protected theorem ClosureAntitone.antitone [TopologicalSpace α] (hA : ClosureAntitone A) :
    CantorScheme.Antitone A := fun l a => subset_closure.trans (hA l a)


protected theorem Antitone.closureAntitone [TopologicalSpace α] (hanti : CantorScheme.Antitone A)
    (hclosed : ∀ l, IsClosed (A l)) : ClosureAntitone A := fun _ _ =>
  (hclosed _).closure_eq.subset.trans (hanti _ _)


/-- A scheme where the children of each set are pairwise disjoint induces an injective map. -/
theorem Disjoint.map_injective (hA : CantorScheme.Disjoint A) : Injective (inducedMap A).2 := by
  /-
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    hA : CantorScheme.Disjoint A
    ⊢ Function.Injective (CantorScheme.inducedMap A).snd
  -/
  rintro ⟨x, hx⟩ ⟨y, hy⟩ hxy
  /-
    case mk.mk
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    hA : CantorScheme.Disjoint A
    x : Nat → β
    hx : Membership.mem (CantorScheme.inducedMap A).fst x
    y : Nat → β
    hy : Membership.mem (CantorScheme.inducedMap A).fst y
    hxy : Eq ((CantorScheme.inducedMap A).snd ⟨x, hx⟩) ((CantorScheme.inducedMap A …
    ⊢ Eq ⟨x, hx⟩ ⟨y, hy⟩
  -/
  refine Subtype.coe_injective (res_injective ?_)
  /-
    case mk.mk
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    hA : CantorScheme.Disjoint A
    x : Nat → β
    hx : Membership.mem (CantorScheme.inducedMap A).fst x
    y : Nat → β
    hy : Membership.mem (CantorScheme.inducedMap A).fst y
    hxy : Eq ((CantorScheme.inducedMap A).snd ⟨x, hx⟩) ((CantorScheme.inducedMap A …
    ⊢ Eq (PiNat.res ((fun a => ↑a) ⟨x, hx⟩)) (PiNat.res ((fun a => ↑a) ⟨y, hy⟩))
  -/
  dsimp
  /-
    case mk.mk
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    hA : CantorScheme.Disjoint A
    x : Nat → β
    hx : Membership.mem (CantorScheme.inducedMap A).fst x
    y : Nat → β
    hy : Membership.mem (CantorScheme.inducedMap A).fst y
    hxy : Eq ((CantorScheme.inducedMap A).snd ⟨x, hx⟩) ((CantorScheme.inducedMap A …
    ⊢ Eq (PiNat.res x) (PiNat.res y)
  -/
  ext n : 1
  /-
    case mk.mk.h
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    hA : CantorScheme.Disjoint A
    x : Nat → β
    hx : Membership.mem (CantorScheme.inducedMap A).fst x
    y : Nat → β
    hy : Membership.mem (CantorScheme.inducedMap A).fst y
    hxy : Eq ((CantorScheme.inducedMap A).snd ⟨x, hx⟩) ((CantorScheme.inducedMap A …
    n : Nat
    ⊢ Eq (PiNat.res x n) (PiNat.res y n)
  -/
  induction' n with n ih; · simp
                            /-
                              🎉 no goals
                            -/
  /-
    case mk.mk.h.succ
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    hA : CantorScheme.Disjoint A
    x : Nat → β
    hx : Membership.mem (CantorScheme.inducedMap A).fst x
    y : Nat → β
    hy : Membership.mem (CantorScheme.inducedMap A).fst y
    hxy : Eq ((CantorScheme.inducedMap A).snd ⟨x, hx⟩) ((CantorScheme.inducedMap A …
    n : Nat
    ih : Eq (PiNat.res x n) (PiNat.res y n)
    ⊢ Eq (PiNat.res x (HAdd.hAdd n 1)) (PiNat.res y (HAdd.hAdd n 1))
  -/
  simp only [res_succ, cons.injEq]
  /-
    case mk.mk.h.succ
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    hA : CantorScheme.Disjoint A
    x : Nat → β
    hx : Membership.mem (CantorScheme.inducedMap A).fst x
    y : Nat → β
    hy : Membership.mem (CantorScheme.inducedMap A).fst y
    hxy : Eq ((CantorScheme.inducedMap A).snd ⟨x, hx⟩) ((CantorScheme.inducedMap A …
    n : Nat
    ih : Eq (PiNat.res x n) (PiNat.res y n)
    ⊢ And (Eq (x n) (y n)) (Eq (PiNat.res x n) (PiNat.res y n))
  -/
  refine ⟨?_, ih⟩
  /-
    case mk.mk.h.succ
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    hA : CantorScheme.Disjoint A
    x : Nat → β
    hx : Membership.mem (CantorScheme.inducedMap A).fst x
    y : Nat → β
    hy : Membership.mem (CantorScheme.inducedMap A).fst y
    hxy : Eq ((CantorScheme.inducedMap A).snd ⟨x, hx⟩) ((CantorScheme.inducedMap A …
    n : Nat
    ih : Eq (PiNat.res x n) (PiNat.res y n)
    ⊢ Eq (x n) (y n)
  -/
  contrapose hA
  /-
    case mk.mk.h.succ
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    x : Nat → β
    hx : Membership.mem (CantorScheme.inducedMap A).fst x
    y : Nat → β
    hy : Membership.mem (CantorScheme.inducedMap A).fst y
    hxy : Eq ((CantorScheme.inducedMap A).snd ⟨x, hx⟩) ((CantorScheme.inducedMap A …
    n : Nat
    ih : Eq (PiNat.res x n) (PiNat.res y n)
    hA : Not (Eq (x n) (y n))
    ⊢ Not (CantorScheme.Disjoint A)
  -/
  simp only [CantorScheme.Disjoint, _root_.Pairwise, Ne, not_forall, exists_prop]
  /-
    case mk.mk.h.succ
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    x : Nat → β
    hx : Membership.mem (CantorScheme.inducedMap A).fst x
    y : Nat → β
    hy : Membership.mem (CantorScheme.inducedMap A).fst y
    hxy : Eq ((CantorScheme.inducedMap A).snd ⟨x, hx⟩) ((CantorScheme.inducedMap A …
    n : Nat
    ih : Eq (PiNat.res x n) (PiNat.res y n)
    hA : Not (Eq (x n) (y n))
    ⊢ Exists fun x => Exists fun x_1 => Exists fun x_2 => And (Not (Eq x_1 x_2)) ( …
  -/
  refine ⟨res x n, _, _, hA, ?_⟩
  /-
    case mk.mk.h.succ
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    x : Nat → β
    hx : Membership.mem (CantorScheme.inducedMap A).fst x
    y : Nat → β
    hy : Membership.mem (CantorScheme.inducedMap A).fst y
    hxy : Eq ((CantorScheme.inducedMap A).snd ⟨x, hx⟩) ((CantorScheme.inducedMap A …
    n : Nat
    ih : Eq (PiNat.res x n) (PiNat.res y n)
    hA : Not (Eq (x n) (y n))
    ⊢ Not (Disjoint (A (List.cons (x n) (PiNat.res x n))) (A (List.cons (y n) (PiN …
  -/
  rw [not_disjoint_iff]
  /-
    case mk.mk.h.succ
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    x : Nat → β
    hx : Membership.mem (CantorScheme.inducedMap A).fst x
    y : Nat → β
    hy : Membership.mem (CantorScheme.inducedMap A).fst y
    hxy : Eq ((CantorScheme.inducedMap A).snd ⟨x, hx⟩) ((CantorScheme.inducedMap A …
    n : Nat
    ih : Eq (PiNat.res x n) (PiNat.res y n)
    hA : Not (Eq (x n) (y n))
    ⊢ Exists fun x_1 => And (Membership.mem (A (List.cons (x n) (PiNat.res x n)))  …
  -/
  refine ⟨(inducedMap A).2 ⟨x, hx⟩, ?_, ?_⟩
    /-
      case mk.mk.h.succ.refine_1
      β : Type u_1
      α : Type u_2
      A : List β → Set α
      x : Nat → β
      hx : Membership.mem (CantorScheme.inducedMap A).fst x
      y : Nat → β
      hy : Membership.mem (CantorScheme.inducedMap A).fst y
      hxy : Eq ((CantorScheme.inducedMap A).snd ⟨x, hx⟩) ((CantorScheme.inducedMap A …
      n : Nat
      ih : Eq (PiNat.res x n) (PiNat.res y n)
      hA : Not (Eq (x n) (y n))
      ⊢ Membership.mem (A (List.cons (x n) (PiNat.res x n))) ((CantorScheme.inducedM …
    -/
  · rw [← res_succ]
    /-
      case mk.mk.h.succ.refine_1
      β : Type u_1
      α : Type u_2
      A : List β → Set α
      x : Nat → β
      hx : Membership.mem (CantorScheme.inducedMap A).fst x
      y : Nat → β
      hy : Membership.mem (CantorScheme.inducedMap A).fst y
      hxy : Eq ((CantorScheme.inducedMap A).snd ⟨x, hx⟩) ((CantorScheme.inducedMap A …
      n : Nat
      ih : Eq (PiNat.res x n) (PiNat.res y n)
      hA : Not (Eq (x n) (y n))
      ⊢ Membership.mem (A (PiNat.res x n.succ)) ((CantorScheme.inducedMap A).snd ⟨x, …
    -/
    apply map_mem
    /-
      🎉 no goals
    -/
  /-
    case mk.mk.h.succ.refine_2
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    x : Nat → β
    hx : Membership.mem (CantorScheme.inducedMap A).fst x
    y : Nat → β
    hy : Membership.mem (CantorScheme.inducedMap A).fst y
    hxy : Eq ((CantorScheme.inducedMap A).snd ⟨x, hx⟩) ((CantorScheme.inducedMap A …
    n : Nat
    ih : Eq (PiNat.res x n) (PiNat.res y n)
    hA : Not (Eq (x n) (y n))
    ⊢ Membership.mem (A (List.cons (y n) (PiNat.res x n))) ((CantorScheme.inducedM …
  -/
  rw [hxy, ih, ← res_succ]
  /-
    case mk.mk.h.succ.refine_2
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    x : Nat → β
    hx : Membership.mem (CantorScheme.inducedMap A).fst x
    y : Nat → β
    hy : Membership.mem (CantorScheme.inducedMap A).fst y
    hxy : Eq ((CantorScheme.inducedMap A).snd ⟨x, hx⟩) ((CantorScheme.inducedMap A …
    n : Nat
    ih : Eq (PiNat.res x n) (PiNat.res y n)
    hA : Not (Eq (x n) (y n))
    ⊢ Membership.mem (A (PiNat.res y n.succ)) ((CantorScheme.inducedMap A).snd ⟨y, …
  -/
  apply map_mem
  /-
    🎉 no goals
  -/


/-- A scheme on a metric space has vanishing diameter if diameter approaches 0 along each branch. -/
def VanishingDiam : Prop :=
  ∀ x : ℕ → β, Tendsto (fun n : ℕ => EMetric.diam (A (res x n))) atTop (𝓝 0)


theorem VanishingDiam.dist_lt (hA : VanishingDiam A) (ε : ℝ) (ε_pos : 0 < ε) (x : ℕ → β) :
    ∃ n : ℕ, ∀ (y) (_ : y ∈ A (res x n)) (z) (_ : z ∈ A (res x n)), dist y z < ε := by
  /-
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    inst✝ : PseudoMetricSpace α
    hA : CantorScheme.VanishingDiam A
    ε : Real
    ε_pos : LT.lt 0 ε
    x : Nat → β
    ⊢ Exists fun n => ∀ (y : α), Membership.mem (A (PiNat.res x n)) y → ∀ (z : α), …
  -/
  specialize hA x
  /-
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    inst✝ : PseudoMetricSpace α
    ε : Real
    ε_pos : LT.lt 0 ε
    x : Nat → β
    hA : Filter.Tendsto (fun n => EMetric.diam (A (PiNat.res x n))) Filter.atTop ( …
    ⊢ Exists fun n => ∀ (y : α), Membership.mem (A (PiNat.res x n)) y → ∀ (z : α), …
  -/
  rw [ENNReal.tendsto_atTop_zero] at hA
  cases' hA (ENNReal.ofReal (ε / 2)) (by
    simp only [gt_iff_lt, ENNReal.ofReal_pos]
    linarith) with n hn
  /-
    case intro
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    inst✝ : PseudoMetricSpace α
    ε : Real
    ε_pos : LT.lt 0 ε
    x : Nat → β
    hA : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LE. …
    n : Nat
    hn : ∀ (n_1 : Nat), GE.ge n_1 n → LE.le (EMetric.diam (A (PiNat.res x n_1))) ( …
    ⊢ Exists fun n => ∀ (y : α), Membership.mem (A (PiNat.res x n)) y → ∀ (z : α), …
  -/
  use n
  /-
    case h
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    inst✝ : PseudoMetricSpace α
    ε : Real
    ε_pos : LT.lt 0 ε
    x : Nat → β
    hA : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LE. …
    n : Nat
    hn : ∀ (n_1 : Nat), GE.ge n_1 n → LE.le (EMetric.diam (A (PiNat.res x n_1))) ( …
    ⊢ ∀ (y : α), Membership.mem (A (PiNat.res x n)) y → ∀ (z : α), Membership.mem  …
  -/
  intro y hy z hz
  /-
    case h
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    inst✝ : PseudoMetricSpace α
    ε : Real
    ε_pos : LT.lt 0 ε
    x : Nat → β
    hA : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LE. …
    n : Nat
    hn : ∀ (n_1 : Nat), GE.ge n_1 n → LE.le (EMetric.diam (A (PiNat.res x n_1))) ( …
    y : α
    hy : Membership.mem (A (PiNat.res x n)) y
    z : α
    hz : Membership.mem (A (PiNat.res x n)) z
    ⊢ LT.lt (Dist.dist y z) ε
  -/
  rw [← ENNReal.ofReal_lt_ofReal_iff ε_pos, ← edist_dist]
  /-
    case h
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    inst✝ : PseudoMetricSpace α
    ε : Real
    ε_pos : LT.lt 0 ε
    x : Nat → β
    hA : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LE. …
    n : Nat
    hn : ∀ (n_1 : Nat), GE.ge n_1 n → LE.le (EMetric.diam (A (PiNat.res x n_1))) ( …
    y : α
    hy : Membership.mem (A (PiNat.res x n)) y
    z : α
    hz : Membership.mem (A (PiNat.res x n)) z
    ⊢ LT.lt (EDist.edist y z) (ENNReal.ofReal ε)
  -/
  apply lt_of_le_of_lt (EMetric.edist_le_diam_of_mem hy hz)
  /-
    case h
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    inst✝ : PseudoMetricSpace α
    ε : Real
    ε_pos : LT.lt 0 ε
    x : Nat → β
    hA : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LE. …
    n : Nat
    hn : ∀ (n_1 : Nat), GE.ge n_1 n → LE.le (EMetric.diam (A (PiNat.res x n_1))) ( …
    y : α
    hy : Membership.mem (A (PiNat.res x n)) y
    z : α
    hz : Membership.mem (A (PiNat.res x n)) z
    ⊢ LT.lt (EMetric.diam (A (PiNat.res x n))) (ENNReal.ofReal ε)
  -/
  apply lt_of_le_of_lt (hn _ (le_refl _))
  /-
    case h
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    inst✝ : PseudoMetricSpace α
    ε : Real
    ε_pos : LT.lt 0 ε
    x : Nat → β
    hA : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LE. …
    n : Nat
    hn : ∀ (n_1 : Nat), GE.ge n_1 n → LE.le (EMetric.diam (A (PiNat.res x n_1))) ( …
    y : α
    hy : Membership.mem (A (PiNat.res x n)) y
    z : α
    hz : Membership.mem (A (PiNat.res x n)) z
    ⊢ LT.lt (ENNReal.ofReal (HDiv.hDiv ε 2)) (ENNReal.ofReal ε)
  -/
  rw [ENNReal.ofReal_lt_ofReal_iff ε_pos]
  /-
    case h
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    inst✝ : PseudoMetricSpace α
    ε : Real
    ε_pos : LT.lt 0 ε
    x : Nat → β
    hA : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LE. …
    n : Nat
    hn : ∀ (n_1 : Nat), GE.ge n_1 n → LE.le (EMetric.diam (A (PiNat.res x n_1))) ( …
    y : α
    hy : Membership.mem (A (PiNat.res x n)) y
    z : α
    hz : Membership.mem (A (PiNat.res x n)) z
    ⊢ LT.lt (HDiv.hDiv ε 2) ε
  -/
  linarith
  /-
    🎉 no goals
  -/


/-- A scheme with vanishing diameter along each branch induces a continuous map. -/
theorem VanishingDiam.map_continuous [TopologicalSpace β] [DiscreteTopology β]
    (hA : VanishingDiam A) : Continuous (inducedMap A).2 := by
  /-
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    inst✝² : PseudoMetricSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : DiscreteTopology β
    hA : CantorScheme.VanishingDiam A
    ⊢ Continuous (CantorScheme.inducedMap A).snd
  -/
  rw [Metric.continuous_iff']
  /-
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    inst✝² : PseudoMetricSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : DiscreteTopology β
    hA : CantorScheme.VanishingDiam A
    ⊢ ∀ (a : ↑(CantorScheme.inducedMap A).fst) (ε : Real), GT.gt ε 0 → Filter.Even …
  -/
  rintro ⟨x, hx⟩ ε ε_pos
  /-
    case mk
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    inst✝² : PseudoMetricSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : DiscreteTopology β
    hA : CantorScheme.VanishingDiam A
    x : Nat → β
    hx : Membership.mem (CantorScheme.inducedMap A).fst x
    ε : Real
    ε_pos : GT.gt ε 0
    ⊢ Filter.Eventually (fun x_1 => LT.lt (Dist.dist ((CantorScheme.inducedMap A). …
  -/
  cases' hA.dist_lt _ ε_pos x with n hn
  /-
    case mk.intro
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    inst✝² : PseudoMetricSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : DiscreteTopology β
    hA : CantorScheme.VanishingDiam A
    x : Nat → β
    hx : Membership.mem (CantorScheme.inducedMap A).fst x
    ε : Real
    ε_pos : GT.gt ε 0
    n : Nat
    hn : ∀ (y : α), Membership.mem (A (PiNat.res x n)) y → ∀ (z : α), Membership.m …
    ⊢ Filter.Eventually (fun x_1 => LT.lt (Dist.dist ((CantorScheme.inducedMap A). …
  -/
  rw [_root_.eventually_nhds_iff]
  /-
    case mk.intro
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    inst✝² : PseudoMetricSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : DiscreteTopology β
    hA : CantorScheme.VanishingDiam A
    x : Nat → β
    hx : Membership.mem (CantorScheme.inducedMap A).fst x
    ε : Real
    ε_pos : GT.gt ε 0
    n : Nat
    hn : ∀ (y : α), Membership.mem (A (PiNat.res x n)) y → ∀ (z : α), Membership.m …
    ⊢ Exists fun t => And (∀ (y : ↑(CantorScheme.inducedMap A).fst), Membership.me …
  -/
  refine ⟨(↑)⁻¹' cylinder x n, ?_, ?_, by simp⟩
    /-
      case mk.intro.refine_1
      β : Type u_1
      α : Type u_2
      A : List β → Set α
      inst✝² : PseudoMetricSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : DiscreteTopology β
      hA : CantorScheme.VanishingDiam A
      x : Nat → β
      hx : Membership.mem (CantorScheme.inducedMap A).fst x
      ε : Real
      ε_pos : GT.gt ε 0
      n : Nat
      hn : ∀ (y : α), Membership.mem (A (PiNat.res x n)) y → ∀ (z : α), Membership.m …
      ⊢ ∀ (y : ↑(CantorScheme.inducedMap A).fst), Membership.mem (Set.preimage Subty …
    -/
  · rintro ⟨y, hy⟩ hyx
    /-
      case mk.intro.refine_1.mk
      β : Type u_1
      α : Type u_2
      A : List β → Set α
      inst✝² : PseudoMetricSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : DiscreteTopology β
      hA : CantorScheme.VanishingDiam A
      x : Nat → β
      hx : Membership.mem (CantorScheme.inducedMap A).fst x
      ε : Real
      ε_pos : GT.gt ε 0
      n : Nat
      hn : ∀ (y : α), Membership.mem (A (PiNat.res x n)) y → ∀ (z : α), Membership.m …
      y : Nat → β
      hy : Membership.mem (CantorScheme.inducedMap A).fst y
      hyx : Membership.mem (Set.preimage Subtype.val (PiNat.cylinder x n)) ⟨y, hy⟩
      ⊢ LT.lt (Dist.dist ((CantorScheme.inducedMap A).snd ⟨y, hy⟩) ((CantorScheme.in …
    -/
    rw [mem_preimage, Subtype.coe_mk, cylinder_eq_res, mem_setOf] at hyx
    /-
      case mk.intro.refine_1.mk
      β : Type u_1
      α : Type u_2
      A : List β → Set α
      inst✝² : PseudoMetricSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : DiscreteTopology β
      hA : CantorScheme.VanishingDiam A
      x : Nat → β
      hx : Membership.mem (CantorScheme.inducedMap A).fst x
      ε : Real
      ε_pos : GT.gt ε 0
      n : Nat
      hn : ∀ (y : α), Membership.mem (A (PiNat.res x n)) y → ∀ (z : α), Membership.m …
      y : Nat → β
      hy : Membership.mem (CantorScheme.inducedMap A).fst y
      hyx : Eq (PiNat.res y n) (PiNat.res x n)
      ⊢ LT.lt (Dist.dist ((CantorScheme.inducedMap A).snd ⟨y, hy⟩) ((CantorScheme.in …
    -/
    apply hn
      /-
        case mk.intro.refine_1.mk.x
        β : Type u_1
        α : Type u_2
        A : List β → Set α
        inst✝² : PseudoMetricSpace α
        inst✝¹ : TopologicalSpace β
        inst✝ : DiscreteTopology β
        hA : CantorScheme.VanishingDiam A
        x : Nat → β
        hx : Membership.mem (CantorScheme.inducedMap A).fst x
        ε : Real
        ε_pos : GT.gt ε 0
        n : Nat
        hn : ∀ (y : α), Membership.mem (A (PiNat.res x n)) y → ∀ (z : α), Membership.m …
        y : Nat → β
        hy : Membership.mem (CantorScheme.inducedMap A).fst y
        hyx : Eq (PiNat.res y n) (PiNat.res x n)
        ⊢ Membership.mem (A (PiNat.res x n)) ((CantorScheme.inducedMap A).snd ⟨y, hy⟩)
      -/
    · rw [← hyx]
      /-
        case mk.intro.refine_1.mk.x
        β : Type u_1
        α : Type u_2
        A : List β → Set α
        inst✝² : PseudoMetricSpace α
        inst✝¹ : TopologicalSpace β
        inst✝ : DiscreteTopology β
        hA : CantorScheme.VanishingDiam A
        x : Nat → β
        hx : Membership.mem (CantorScheme.inducedMap A).fst x
        ε : Real
        ε_pos : GT.gt ε 0
        n : Nat
        hn : ∀ (y : α), Membership.mem (A (PiNat.res x n)) y → ∀ (z : α), Membership.m …
        y : Nat → β
        hy : Membership.mem (CantorScheme.inducedMap A).fst y
        hyx : Eq (PiNat.res y n) (PiNat.res x n)
        ⊢ Membership.mem (A (PiNat.res y n)) ((CantorScheme.inducedMap A).snd ⟨y, hy⟩)
      -/
      apply map_mem
      /-
        🎉 no goals
      -/
    /-
      case mk.intro.refine_1.mk.x
      β : Type u_1
      α : Type u_2
      A : List β → Set α
      inst✝² : PseudoMetricSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : DiscreteTopology β
      hA : CantorScheme.VanishingDiam A
      x : Nat → β
      hx : Membership.mem (CantorScheme.inducedMap A).fst x
      ε : Real
      ε_pos : GT.gt ε 0
      n : Nat
      hn : ∀ (y : α), Membership.mem (A (PiNat.res x n)) y → ∀ (z : α), Membership.m …
      y : Nat → β
      hy : Membership.mem (CantorScheme.inducedMap A).fst y
      hyx : Eq (PiNat.res y n) (PiNat.res x n)
      ⊢ Membership.mem (A (PiNat.res x n)) ((CantorScheme.inducedMap A).snd ⟨x, hx⟩)
    -/
    apply map_mem
    /-
      🎉 no goals
    -/
  /-
    case mk.intro.refine_2
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    inst✝² : PseudoMetricSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : DiscreteTopology β
    hA : CantorScheme.VanishingDiam A
    x : Nat → β
    hx : Membership.mem (CantorScheme.inducedMap A).fst x
    ε : Real
    ε_pos : GT.gt ε 0
    n : Nat
    hn : ∀ (y : α), Membership.mem (A (PiNat.res x n)) y → ∀ (z : α), Membership.m …
    ⊢ IsOpen (Set.preimage Subtype.val (PiNat.cylinder x n))
  -/
  apply continuous_subtype_val.isOpen_preimage
  /-
    case mk.intro.refine_2.a
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    inst✝² : PseudoMetricSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : DiscreteTopology β
    hA : CantorScheme.VanishingDiam A
    x : Nat → β
    hx : Membership.mem (CantorScheme.inducedMap A).fst x
    ε : Real
    ε_pos : GT.gt ε 0
    n : Nat
    hn : ∀ (y : α), Membership.mem (A (PiNat.res x n)) y → ∀ (z : α), Membership.m …
    ⊢ IsOpen (PiNat.cylinder x n)
  -/
  apply isOpen_cylinder
  /-
    🎉 no goals
  -/


/-- A scheme on a complete space with vanishing diameter
such that each set contains the closure of its children
induces a total map. -/
theorem ClosureAntitone.map_of_vanishingDiam [CompleteSpace α] (hdiam : VanishingDiam A)
    (hanti : ClosureAntitone A) (hnonempty : ∀ l, (A l).Nonempty) : (inducedMap A).1 = univ := by
  /-
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    inst✝¹ : PseudoMetricSpace α
    inst✝ : CompleteSpace α
    hdiam : CantorScheme.VanishingDiam A
    hanti : CantorScheme.ClosureAntitone A
    hnonempty : ∀ (l : List β), (A l).Nonempty
    ⊢ Eq (CantorScheme.inducedMap A).fst Set.univ
  -/
  rw [eq_univ_iff_forall]
  /-
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    inst✝¹ : PseudoMetricSpace α
    inst✝ : CompleteSpace α
    hdiam : CantorScheme.VanishingDiam A
    hanti : CantorScheme.ClosureAntitone A
    hnonempty : ∀ (l : List β), (A l).Nonempty
    ⊢ ∀ (x : Nat → β), Membership.mem (CantorScheme.inducedMap A).fst x
  -/
  intro x
  /-
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    inst✝¹ : PseudoMetricSpace α
    inst✝ : CompleteSpace α
    hdiam : CantorScheme.VanishingDiam A
    hanti : CantorScheme.ClosureAntitone A
    hnonempty : ∀ (l : List β), (A l).Nonempty
    x : Nat → β
    ⊢ Membership.mem (CantorScheme.inducedMap A).fst x
  -/
  choose u hu using fun n => hnonempty (res x n)
  have umem : ∀ n m : ℕ, n ≤ m → u m ∈ A (res x n) := by
    have : Antitone fun n : ℕ => A (res x n) := by
      refine antitone_nat_of_succ_le ?_
      intro n
      apply hanti.antitone
    intro n m hnm
    exact this hnm (hu _)
  have : CauchySeq u := by
    rw [Metric.cauchySeq_iff]
    intro ε ε_pos
    cases' hdiam.dist_lt _ ε_pos x with n hn
    use n
    intro m₀ hm₀ m₁ hm₁
    apply hn <;> apply umem <;> assumption
  /-
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    inst✝¹ : PseudoMetricSpace α
    inst✝ : CompleteSpace α
    hdiam : CantorScheme.VanishingDiam A
    hanti : CantorScheme.ClosureAntitone A
    hnonempty : ∀ (l : List β), (A l).Nonempty
    x : Nat → β
    u : Nat → α
    hu : ∀ (n : Nat), Membership.mem (A (PiNat.res x n)) (u n)
    umem : ∀ (n m : Nat), LE.le n m → Membership.mem (A (PiNat.res x n)) (u m)
    this : CauchySeq u
    ⊢ Membership.mem (CantorScheme.inducedMap A).fst x
  -/
  cases' cauchySeq_tendsto_of_complete this with y hy
  /-
    case intro
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    inst✝¹ : PseudoMetricSpace α
    inst✝ : CompleteSpace α
    hdiam : CantorScheme.VanishingDiam A
    hanti : CantorScheme.ClosureAntitone A
    hnonempty : ∀ (l : List β), (A l).Nonempty
    x : Nat → β
    u : Nat → α
    hu : ∀ (n : Nat), Membership.mem (A (PiNat.res x n)) (u n)
    umem : ∀ (n m : Nat), LE.le n m → Membership.mem (A (PiNat.res x n)) (u m)
    this : CauchySeq u
    y : α
    hy : Filter.Tendsto u Filter.atTop (nhds y)
    ⊢ Membership.mem (CantorScheme.inducedMap A).fst x
  -/
  use y
  /-
    case h
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    inst✝¹ : PseudoMetricSpace α
    inst✝ : CompleteSpace α
    hdiam : CantorScheme.VanishingDiam A
    hanti : CantorScheme.ClosureAntitone A
    hnonempty : ∀ (l : List β), (A l).Nonempty
    x : Nat → β
    u : Nat → α
    hu : ∀ (n : Nat), Membership.mem (A (PiNat.res x n)) (u n)
    umem : ∀ (n m : Nat), LE.le n m → Membership.mem (A (PiNat.res x n)) (u m)
    this : CauchySeq u
    y : α
    hy : Filter.Tendsto u Filter.atTop (nhds y)
    ⊢ Membership.mem (Set.iInter fun n => A (PiNat.res x n)) y
  -/
  rw [mem_iInter]
  /-
    case h
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    inst✝¹ : PseudoMetricSpace α
    inst✝ : CompleteSpace α
    hdiam : CantorScheme.VanishingDiam A
    hanti : CantorScheme.ClosureAntitone A
    hnonempty : ∀ (l : List β), (A l).Nonempty
    x : Nat → β
    u : Nat → α
    hu : ∀ (n : Nat), Membership.mem (A (PiNat.res x n)) (u n)
    umem : ∀ (n m : Nat), LE.le n m → Membership.mem (A (PiNat.res x n)) (u m)
    this : CauchySeq u
    y : α
    hy : Filter.Tendsto u Filter.atTop (nhds y)
    ⊢ ∀ (i : Nat), Membership.mem (A (PiNat.res x i)) y
  -/
  intro n
  /-
    case h
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    inst✝¹ : PseudoMetricSpace α
    inst✝ : CompleteSpace α
    hdiam : CantorScheme.VanishingDiam A
    hanti : CantorScheme.ClosureAntitone A
    hnonempty : ∀ (l : List β), (A l).Nonempty
    x : Nat → β
    u : Nat → α
    hu : ∀ (n : Nat), Membership.mem (A (PiNat.res x n)) (u n)
    umem : ∀ (n m : Nat), LE.le n m → Membership.mem (A (PiNat.res x n)) (u m)
    this : CauchySeq u
    y : α
    hy : Filter.Tendsto u Filter.atTop (nhds y)
    n : Nat
    ⊢ Membership.mem (A (PiNat.res x n)) y
  -/
  apply hanti _ (x n)
  /-
    case h.a
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    inst✝¹ : PseudoMetricSpace α
    inst✝ : CompleteSpace α
    hdiam : CantorScheme.VanishingDiam A
    hanti : CantorScheme.ClosureAntitone A
    hnonempty : ∀ (l : List β), (A l).Nonempty
    x : Nat → β
    u : Nat → α
    hu : ∀ (n : Nat), Membership.mem (A (PiNat.res x n)) (u n)
    umem : ∀ (n m : Nat), LE.le n m → Membership.mem (A (PiNat.res x n)) (u m)
    this : CauchySeq u
    y : α
    hy : Filter.Tendsto u Filter.atTop (nhds y)
    n : Nat
    ⊢ Membership.mem (closure (A (List.cons (x n) (PiNat.res x n)))) y
  -/
  apply mem_closure_of_tendsto hy
  /-
    case h.a
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    inst✝¹ : PseudoMetricSpace α
    inst✝ : CompleteSpace α
    hdiam : CantorScheme.VanishingDiam A
    hanti : CantorScheme.ClosureAntitone A
    hnonempty : ∀ (l : List β), (A l).Nonempty
    x : Nat → β
    u : Nat → α
    hu : ∀ (n : Nat), Membership.mem (A (PiNat.res x n)) (u n)
    umem : ∀ (n m : Nat), LE.le n m → Membership.mem (A (PiNat.res x n)) (u m)
    this : CauchySeq u
    y : α
    hy : Filter.Tendsto u Filter.atTop (nhds y)
    n : Nat
    ⊢ Filter.Eventually (fun x_1 => Membership.mem (A (List.cons (x n) (PiNat.res  …
  -/
  rw [eventually_atTop]
  /-
    case h.a
    β : Type u_1
    α : Type u_2
    A : List β → Set α
    inst✝¹ : PseudoMetricSpace α
    inst✝ : CompleteSpace α
    hdiam : CantorScheme.VanishingDiam A
    hanti : CantorScheme.ClosureAntitone A
    hnonempty : ∀ (l : List β), (A l).Nonempty
    x : Nat → β
    u : Nat → α
    hu : ∀ (n : Nat), Membership.mem (A (PiNat.res x n)) (u n)
    umem : ∀ (n m : Nat), LE.le n m → Membership.mem (A (PiNat.res x n)) (u m)
    this : CauchySeq u
    y : α
    hy : Filter.Tendsto u Filter.atTop (nhds y)
    n : Nat
    ⊢ Exists fun a => ∀ (b : Nat), GE.ge b a → Membership.mem (A (List.cons (x n)  …
  -/
  exact ⟨n.succ, umem _⟩
  /-
    🎉 no goals
  -/


