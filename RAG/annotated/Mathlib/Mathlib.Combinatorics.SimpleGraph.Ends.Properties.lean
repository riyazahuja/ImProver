instance [Finite V] : IsEmpty G.end where
  false := by
    /-
      V : Type
      G : SimpleGraph V
      inst✝ : Finite V
      ⊢ ↑G.end → False
    -/
    rintro ⟨s, _⟩
    /-
      case mk
      V : Type
      G : SimpleGraph V
      inst✝ : Finite V
      s : (j : Opposite (Finset V)) → G.componentComplFunctor.obj j
      property✝ : Membership.mem G.end s
      ⊢ False
    -/
    cases nonempty_fintype V
    /-
      case mk.intro
      V : Type
      G : SimpleGraph V
      inst✝ : Finite V
      s : (j : Opposite (Finset V)) → G.componentComplFunctor.obj j
      property✝ : Membership.mem G.end s
      val✝ : Fintype V
      ⊢ False
    -/
    obtain ⟨v, h⟩ := (s <| Opposite.op Finset.univ).nonempty
    exact Set.disjoint_iff.mp (s _).disjoint_right
        ⟨by simp only [Opposite.unop_op, Finset.coe_univ, Set.mem_univ], h⟩


/-- The `componentCompl`s chosen by an end are all infinite. -/
lemma end_componentCompl_infinite (e : G.end) (K : (Finset V)ᵒᵖ) :
    ((e : (j : (Finset V)ᵒᵖ) → G.componentComplFunctor.obj j) K).supp.Infinite := by
  /-
    V : Type
    G : SimpleGraph V
    e : ↑G.end
    K : Opposite (Finset V)
    ⊢ (SimpleGraph.ComponentCompl.supp (↑e K)).Infinite
  -/
  refine (e.val K).infinite_iff_in_all_ranges.mpr (fun L h => ?_)
  /-
    V : Type
    G : SimpleGraph V
    e : ↑G.end
    K : Opposite (Finset V)
    L : Finset V
    h : HasSubset.Subset (Opposite.unop K) L
    ⊢ Exists fun D => Eq (SimpleGraph.ComponentCompl.hom h D) (↑e K)
  -/
  change Opposite.unop K ⊆ Opposite.unop (Opposite.op L) at h
  /-
    V : Type
    G : SimpleGraph V
    e : ↑G.end
    K : Opposite (Finset V)
    L : Finset V
    h : HasSubset.Subset (Opposite.unop K) (Opposite.unop { unop := L })
    ⊢ Exists fun D => Eq (SimpleGraph.ComponentCompl.hom h D) (↑e K)
  -/
  exact ⟨e.val (Opposite.op L), (e.prop (CategoryTheory.opHomOfLE h))⟩
  /-
    🎉 no goals
  -/


instance compononentComplFunctor_nonempty_of_infinite [Infinite V] (K : (Finset V)ᵒᵖ) :
    Nonempty (G.componentComplFunctor.obj K) := G.componentCompl_nonempty_of_infinite K.unop


instance componentComplFunctor_finite [LocallyFinite G] [Fact G.Preconnected]
    (K : (Finset V)ᵒᵖ) : Finite (G.componentComplFunctor.obj K) := G.componentCompl_finite K.unop


/-- A locally finite preconnected infinite graph has at least one end. -/
lemma nonempty_ends_of_infinite [LocallyFinite G] [Fact G.Preconnected] [Infinite V] :
    G.end.Nonempty := by
  classical
  apply nonempty_sections_of_finite_inverse_system G.componentComplFunctor


