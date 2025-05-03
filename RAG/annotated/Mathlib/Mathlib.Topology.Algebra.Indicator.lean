@[to_additive]
lemma continuous_mulIndicator (hs : ∀ a ∈ frontier s, f a = 1) (hf : ContinuousOn f (closure s)) :
    Continuous (mulIndicator s f) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    f : α → β
    s : Set α
    inst✝ : One β
    hs : ∀ (a : α), Membership.mem (frontier s) a → Eq (f a) 1
    hf : ContinuousOn f (closure s)
    ⊢ Continuous (s.mulIndicator f)
  -/
  classical exact continuous_piecewise hs hf continuousOn_const
  /-
    🎉 no goals
  -/


@[to_additive]
protected lemma Continuous.mulIndicator (hs : ∀ a ∈ frontier s, f a = 1) (hf : Continuous f) :
    Continuous (mulIndicator s f) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    f : α → β
    s : Set α
    inst✝ : One β
    hs : ∀ (a : α), Membership.mem (frontier s) a → Eq (f a) 1
    hf : Continuous f
    ⊢ Continuous (s.mulIndicator f)
  -/
  classical exact hf.piecewise hs continuous_const
  /-
    🎉 no goals
  -/


@[to_additive]
theorem ContinuousOn.continuousAt_mulIndicator (hf : ContinuousOn f (interior s)) {x : α}
    (hx : x ∉ frontier s) :
    ContinuousAt (s.mulIndicator f) x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    f : α → β
    s : Set α
    inst✝ : One β
    hf : ContinuousOn f (interior s)
    x : α
    hx : Not (Membership.mem (frontier s) x)
    ⊢ ContinuousAt (s.mulIndicator f) x
  -/
  rw [← Set.mem_compl_iff, compl_frontier_eq_union_interior] at hx
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    f : α → β
    s : Set α
    inst✝ : One β
    hf : ContinuousOn f (interior s)
    x : α
    hx : Membership.mem (Union.union (interior s) (interior (HasCompl.compl s))) x
    ⊢ ContinuousAt (s.mulIndicator f) x
  -/
  obtain h | h := hx
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f : α → β
      s : Set α
      inst✝ : One β
      hf : ContinuousOn f (interior s)
      x : α
      h : Membership.mem (interior s) x
      ⊢ ContinuousAt (s.mulIndicator f) x
    -/
  · have hs : interior s ∈ 𝓝 x := mem_interior_iff_mem_nhds.mp (by rwa [interior_interior])
    exact ContinuousAt.congr (hf.continuousAt hs) <| Filter.eventuallyEq_iff_exists_mem.mpr
      ⟨interior s, hs, Set.eqOn_mulIndicator.symm.mono interior_subset⟩
  · exact ContinuousAt.congr continuousAt_const <| Filter.eventuallyEq_iff_exists_mem.mpr
      ⟨sᶜ, mem_interior_iff_mem_nhds.mp h, Set.eqOn_mulIndicator'.symm⟩

