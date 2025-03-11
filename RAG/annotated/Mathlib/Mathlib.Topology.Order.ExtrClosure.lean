protected theorem IsMaxOn.closure (h : IsMaxOn f s a) (hc : ContinuousOn f (closure s)) :
    IsMaxOn f (closure s) a := fun x hx =>
  ContinuousWithinAt.closure_le hx ((hc x hx).mono subset_closure) continuousWithinAt_const h


protected theorem IsMinOn.closure (h : IsMinOn f s a) (hc : ContinuousOn f (closure s)) :
    IsMinOn f (closure s) a :=
  h.dual.closure hc


protected theorem IsExtrOn.closure (h : IsExtrOn f s a) (hc : ContinuousOn f (closure s)) :
    IsExtrOn f (closure s) a :=
  h.elim (fun h => Or.inl <| h.closure hc) fun h => Or.inr <| h.closure hc


protected theorem IsLocalMaxOn.closure (h : IsLocalMaxOn f s a) (hc : ContinuousOn f (closure s)) :
    IsLocalMaxOn f (closure s) a := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : Preorder Y
    inst✝ : OrderClosedTopology Y
    f : X → Y
    s : Set X
    a : X
    h : IsLocalMaxOn f s a
    hc : ContinuousOn f (closure s)
    ⊢ IsLocalMaxOn f (closure s) a
  -/
  rcases mem_nhdsWithin.1 h with ⟨U, Uo, aU, hU⟩
  /-
    case intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : Preorder Y
    inst✝ : OrderClosedTopology Y
    f : X → Y
    s : Set X
    a : X
    h : IsLocalMaxOn f s a
    hc : ContinuousOn f (closure s)
    U : Set X
    Uo : IsOpen U
    aU : Membership.mem U a
    hU : HasSubset.Subset (Inter.inter U s) (setOf fun x => (fun x => LE.le (f x)  …
    ⊢ IsLocalMaxOn f (closure s) a
  -/
  refine mem_nhdsWithin.2 ⟨U, Uo, aU, ?_⟩
  /-
    case intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : Preorder Y
    inst✝ : OrderClosedTopology Y
    f : X → Y
    s : Set X
    a : X
    h : IsLocalMaxOn f s a
    hc : ContinuousOn f (closure s)
    U : Set X
    Uo : IsOpen U
    aU : Membership.mem U a
    hU : HasSubset.Subset (Inter.inter U s) (setOf fun x => (fun x => LE.le (f x)  …
    ⊢ HasSubset.Subset (Inter.inter U (closure s)) (setOf fun x => (fun x => LE.le …
  -/
  rintro x ⟨hxU, hxs⟩
  /-
    case intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : Preorder Y
    inst✝ : OrderClosedTopology Y
    f : X → Y
    s : Set X
    a : X
    h : IsLocalMaxOn f s a
    hc : ContinuousOn f (closure s)
    U : Set X
    Uo : IsOpen U
    aU : Membership.mem U a
    hU : HasSubset.Subset (Inter.inter U s) (setOf fun x => (fun x => LE.le (f x)  …
    x : X
    hxU : Membership.mem U x
    hxs : Membership.mem (closure s) x
    ⊢ Membership.mem (setOf fun x => (fun x => LE.le (f x) (f a)) x) x
  -/
  refine ContinuousWithinAt.closure_le ?_ ?_ continuousWithinAt_const hU
  · rwa [mem_closure_iff_nhdsWithin_neBot, nhdsWithin_inter_of_mem, ←
      mem_closure_iff_nhdsWithin_neBot]
    /-
      case intro.intro.intro.intro.refine_1
      X : Type u_1
      Y : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : Preorder Y
      inst✝ : OrderClosedTopology Y
      f : X → Y
      s : Set X
      a : X
      h : IsLocalMaxOn f s a
      hc : ContinuousOn f (closure s)
      U : Set X
      Uo : IsOpen U
      aU : Membership.mem U a
      hU : HasSubset.Subset (Inter.inter U s) (setOf fun x => (fun x => LE.le (f x)  …
      x : X
      hxU : Membership.mem U x
      hxs : Membership.mem (closure s) x
      ⊢ Membership.mem (nhdsWithin x s) U
    -/
    exact nhdsWithin_le_nhds (Uo.mem_nhds hxU)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_2
      X : Type u_1
      Y : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : Preorder Y
      inst✝ : OrderClosedTopology Y
      f : X → Y
      s : Set X
      a : X
      h : IsLocalMaxOn f s a
      hc : ContinuousOn f (closure s)
      U : Set X
      Uo : IsOpen U
      aU : Membership.mem U a
      hU : HasSubset.Subset (Inter.inter U s) (setOf fun x => (fun x => LE.le (f x)  …
      x : X
      hxU : Membership.mem U x
      hxs : Membership.mem (closure s) x
      ⊢ ContinuousWithinAt f (Inter.inter U s) x
    -/
  · exact (hc _ hxs).mono (inter_subset_right.trans subset_closure)
    /-
      🎉 no goals
    -/


protected theorem IsLocalMinOn.closure (h : IsLocalMinOn f s a) (hc : ContinuousOn f (closure s)) :
    IsLocalMinOn f (closure s) a :=
  IsLocalMaxOn.closure h.dual hc


protected theorem IsLocalExtrOn.closure (h : IsLocalExtrOn f s a)
    (hc : ContinuousOn f (closure s)) : IsLocalExtrOn f (closure s) a :=
  h.elim (fun h => Or.inl <| h.closure hc) fun h => Or.inr <| h.closure hc

