/-- A point `x : X` is evenly covered by `f : E → X` if `x` has an evenly covered neighborhood. -/
def IsEvenlyCovered (x : X) (I : Type*) [TopologicalSpace I] :=
  DiscreteTopology I ∧ ∃ t : Trivialization I f, x ∈ t.baseSet


/-- If `x` is evenly covered by `f`, then we can construct a trivialization of `f` at `x`. -/
noncomputable def toTrivialization {x : X} {I : Type*} [TopologicalSpace I]
    (h : IsEvenlyCovered f x I) : Trivialization (f ⁻¹' {x}) f :=
  (Classical.choose h.2).transFiberHomeomorph
    ((Classical.choose h.2).preimageSingletonHomeomorph (Classical.choose_spec h.2)).symm


theorem mem_toTrivialization_baseSet {x : X} {I : Type*} [TopologicalSpace I]
    (h : IsEvenlyCovered f x I) : x ∈ h.toTrivialization.baseSet :=
  Classical.choose_spec h.2


theorem toTrivialization_apply {x : E} {I : Type*} [TopologicalSpace I]
    (h : IsEvenlyCovered f (f x) I) : (h.toTrivialization x).2 = ⟨x, rfl⟩ :=
  let e := Classical.choose h.2
  let h := Classical.choose_spec h.2
  let he := e.mk_proj_snd' h
  Subtype.ext
    ((e.toPartialEquiv.eq_symm_apply (e.mem_source.mpr h)
                /-
                  E : Type u_1
                  X : Type u_2
                  inst✝² : TopologicalSpace E
                  inst✝¹ : TopologicalSpace X
                  f : E → X
                  x : E
                  I : Type u_3
                  inst✝ : TopologicalSpace I
                  h✝ : IsEvenlyCovered f (f x) I
                  e : Trivialization I f := Classical.choose ⋯
                  h : Membership.mem (Classical.choose ⋯).baseSet (f x) := Classical.choose_spec …
                  he : Eq { fst := f x, snd := (↑e x).2 } (↑e x) := Trivialization.mk_proj_snd'  …
                  ⊢ Membership.mem e.target { fst := f x, snd := (↑e x).2 }
                -/
            (by rwa [he, e.mem_target, e.coe_fst (e.mem_source.mpr h)])).mpr
                /-
                  🎉 no goals
                -/
        he.symm).symm


protected theorem continuousAt {x : E} {I : Type*} [TopologicalSpace I]
    (h : IsEvenlyCovered f (f x) I) : ContinuousAt f x :=
  let e := h.toTrivialization
  e.continuousAt_proj (e.mem_source.mpr (mem_toTrivialization_baseSet h))


theorem to_isEvenlyCovered_preimage {x : X} {I : Type*} [TopologicalSpace I]
    (h : IsEvenlyCovered f x I) : IsEvenlyCovered f x (f ⁻¹' {x}) :=
  let ⟨_, h2⟩ := h
  ⟨((Classical.choose h2).preimageSingletonHomeomorph
          (Classical.choose_spec h2)).isEmbedding.discreteTopology,
    _, h.mem_toTrivialization_baseSet⟩


/-- A covering map is a continuous function `f : E → X` with discrete fibers such that each point
  of `X` has an evenly covered neighborhood. -/
def IsCoveringMapOn :=
  ∀ x ∈ s, IsEvenlyCovered f x (f ⁻¹' {x})


theorem mk (F : X → Type*) [∀ x, TopologicalSpace (F x)] [hF : ∀ x, DiscreteTopology (F x)]
    (e : ∀ x ∈ s, Trivialization (F x) f) (h : ∀ (x : X) (hx : x ∈ s), x ∈ (e x hx).baseSet) :
    IsCoveringMapOn f s := fun x hx =>
  IsEvenlyCovered.to_isEvenlyCovered_preimage ⟨hF x, e x hx, h x hx⟩


protected theorem continuousAt (hf : IsCoveringMapOn f s) {x : E} (hx : f x ∈ s) :
    ContinuousAt f x :=
  (hf (f x) hx).continuousAt


protected theorem continuousOn (hf : IsCoveringMapOn f s) : ContinuousOn f (f ⁻¹' s) :=
  continuousOn_of_forall_continuousAt fun _ => hf.continuousAt


protected theorem isLocalHomeomorphOn (hf : IsCoveringMapOn f s) :
    IsLocalHomeomorphOn f (f ⁻¹' s) := by
  /-
    E : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalSpace X
    f : E → X
    s : Set X
    hf : IsCoveringMapOn f s
    ⊢ IsLocalHomeomorphOn f (Set.preimage f s)
  -/
  refine IsLocalHomeomorphOn.mk f (f ⁻¹' s) fun x hx => ?_
  /-
    E : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalSpace X
    f : E → X
    s : Set X
    hf : IsCoveringMapOn f s
    x : E
    hx : Membership.mem (Set.preimage f s) x
    ⊢ Exists fun e => And (Membership.mem e.source x) (Set.EqOn f (↑e) e.source)
  -/
  let e := (hf (f x) hx).toTrivialization
  /-
    E : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalSpace X
    f : E → X
    s : Set X
    hf : IsCoveringMapOn f s
    x : E
    hx : Membership.mem (Set.preimage f s) x
    e : Trivialization (↑(Set.preimage f (Singleton.singleton (f x)))) f := ⋯.toTr …
    ⊢ Exists fun e => And (Membership.mem e.source x) (Set.EqOn f (↑e) e.source)
  -/
  have h := (hf (f x) hx).mem_toTrivialization_baseSet
  /-
    E : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalSpace X
    f : E → X
    s : Set X
    hf : IsCoveringMapOn f s
    x : E
    hx : Membership.mem (Set.preimage f s) x
    e : Trivialization (↑(Set.preimage f (Singleton.singleton (f x)))) f := ⋯.toTr …
    h : Membership.mem ⋯.toTrivialization.baseSet (f x)
    ⊢ Exists fun e => And (Membership.mem e.source x) (Set.EqOn f (↑e) e.source)
  -/
  let he := e.mem_source.2 h
  refine
    ⟨e.toPartialHomeomorph.trans
        { toFun := fun p => p.1
          invFun := fun p => ⟨p, x, rfl⟩
          source := e.baseSet ×ˢ ({⟨x, rfl⟩} : Set (f ⁻¹' {f x}))
          target := e.baseSet
          open_source :=
            e.open_baseSet.prod (singletons_open_iff_discrete.2 (hf (f x) hx).1 ⟨x, rfl⟩)
          open_target := e.open_baseSet
          map_source' := fun p => And.left
          map_target' := fun p hp => ⟨hp, rfl⟩
          left_inv' := fun p hp => Prod.ext rfl hp.2.symm
          right_inv' := fun p _ => rfl
          continuousOn_toFun := continuous_fst.continuousOn
          continuousOn_invFun := (continuous_id'.prod_mk continuous_const).continuousOn },
      ⟨he, by rwa [e.toPartialHomeomorph.symm_symm, e.proj_toFun x he],
        (hf (f x) hx).toTrivialization_apply⟩,
      fun p h => (e.proj_toFun p h.1).symm⟩


/-- A covering map is a continuous function `f : E → X` with discrete fibers such that each point
  of `X` has an evenly covered neighborhood. -/
def IsCoveringMap :=
  ∀ x, IsEvenlyCovered f x (f ⁻¹' {x})


theorem isCoveringMap_iff_isCoveringMapOn_univ : IsCoveringMap f ↔ IsCoveringMapOn f Set.univ := by
  /-
    E : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalSpace X
    f : E → X
    ⊢ Iff (IsCoveringMap f) (IsCoveringMapOn f Set.univ)
  -/
  simp only [IsCoveringMap, IsCoveringMapOn, Set.mem_univ, forall_true_left]
  /-
    🎉 no goals
  -/


protected theorem IsCoveringMap.isCoveringMapOn (hf : IsCoveringMap f) :
    IsCoveringMapOn f Set.univ :=
  isCoveringMap_iff_isCoveringMapOn_univ.mp hf


theorem mk (F : X → Type*) [∀ x, TopologicalSpace (F x)] [∀ x, DiscreteTopology (F x)]
    (e : ∀ x, Trivialization (F x) f) (h : ∀ x, x ∈ (e x).baseSet) : IsCoveringMap f :=
  isCoveringMap_iff_isCoveringMapOn_univ.mpr
    (IsCoveringMapOn.mk f Set.univ F (fun x _ => e x) fun x _ => h x)


protected theorem continuous : Continuous f :=
  continuous_iff_continuousOn_univ.mpr hf.isCoveringMapOn.continuousOn


protected theorem isLocalHomeomorph : IsLocalHomeomorph f :=
  isLocalHomeomorph_iff_isLocalHomeomorphOn_univ.mpr hf.isCoveringMapOn.isLocalHomeomorphOn


protected theorem isOpenMap : IsOpenMap f :=
  hf.isLocalHomeomorph.isOpenMap


theorem isQuotientMap (hf' : Function.Surjective f) : IsQuotientMap f :=
  hf.isOpenMap.isQuotientMap hf.continuous hf'


@[deprecated (since := "2024-10-22")]
alias quotientMap := isQuotientMap


protected theorem isSeparatedMap : IsSeparatedMap f :=
  fun e₁ e₂ he hne ↦ by
    /-
      E : Type u_1
      X : Type u_2
      inst✝¹ : TopologicalSpace E
      inst✝ : TopologicalSpace X
      f : E → X
      hf : IsCoveringMap f
      e₁ e₂ : E
      he : Eq (f e₁) (f e₂)
      hne : Ne e₁ e₂
      ⊢ Exists fun s₁ => Exists fun s₂ => And (IsOpen s₁) (And (IsOpen s₂) (And (Mem …
    -/
    obtain ⟨_, t, he₁⟩ := hf (f e₁)
    /-
      case intro.intro
      E : Type u_1
      X : Type u_2
      inst✝¹ : TopologicalSpace E
      inst✝ : TopologicalSpace X
      f : E → X
      hf : IsCoveringMap f
      e₁ e₂ : E
      he : Eq (f e₁) (f e₂)
      hne : Ne e₁ e₂
      left✝ : DiscreteTopology ↑(Set.preimage f (Singleton.singleton (f e₁)))
      t : Trivialization (↑(Set.preimage f (Singleton.singleton (f e₁)))) f
      he₁ : Membership.mem t.baseSet (f e₁)
      ⊢ Exists fun s₁ => Exists fun s₂ => And (IsOpen s₁) (And (IsOpen s₂) (And (Mem …
    -/
    have he₂ := he₁; simp_rw [he] at he₂; rw [← t.mem_source] at he₁ he₂
    refine ⟨t.source ∩ (Prod.snd ∘ t) ⁻¹' {(t e₁).2}, t.source ∩ (Prod.snd ∘ t) ⁻¹' {(t e₂).2},
      ?_, ?_, ⟨he₁, rfl⟩, ⟨he₂, rfl⟩, Set.disjoint_left.mpr fun x h₁ h₂ ↦ hne (t.injOn he₁ he₂ ?_)⟩
    iterate 2
      exact t.continuousOn_toFun.isOpen_inter_preimage t.open_source
        (continuous_snd.isOpen_preimage _ <| isOpen_discrete _)
    /-
      case intro.intro.refine_3
      E : Type u_1
      X : Type u_2
      inst✝¹ : TopologicalSpace E
      inst✝ : TopologicalSpace X
      f : E → X
      hf : IsCoveringMap f
      e₁ e₂ : E
      he : Eq (f e₁) (f e₂)
      hne : Ne e₁ e₂
      left✝ : DiscreteTopology ↑(Set.preimage f (Singleton.singleton (f e₁)))
      t : Trivialization (↑(Set.preimage f (Singleton.singleton (f e₁)))) f
      he₁ : Membership.mem t.source e₁
      he₂ : Membership.mem t.source e₂
      x : E
      h₁ : Membership.mem (Inter.inter t.source (Set.preimage (Function.comp Prod.sn …
      h₂ : Membership.mem (Inter.inter t.source (Set.preimage (Function.comp Prod.sn …
      ⊢ Eq (↑t.toPartialHomeomorph e₁) (↑t.toPartialHomeomorph e₂)
    -/
    refine Prod.ext ?_ (h₁.2.symm.trans h₂.2)
    /-
      case intro.intro.refine_3
      E : Type u_1
      X : Type u_2
      inst✝¹ : TopologicalSpace E
      inst✝ : TopologicalSpace X
      f : E → X
      hf : IsCoveringMap f
      e₁ e₂ : E
      he : Eq (f e₁) (f e₂)
      hne : Ne e₁ e₂
      left✝ : DiscreteTopology ↑(Set.preimage f (Singleton.singleton (f e₁)))
      t : Trivialization (↑(Set.preimage f (Singleton.singleton (f e₁)))) f
      he₁ : Membership.mem t.source e₁
      he₂ : Membership.mem t.source e₂
      x : E
      h₁ : Membership.mem (Inter.inter t.source (Set.preimage (Function.comp Prod.sn …
      h₂ : Membership.mem (Inter.inter t.source (Set.preimage (Function.comp Prod.sn …
      ⊢ Eq (↑t.toPartialHomeomorph e₁).1 (↑t.toPartialHomeomorph e₂).1
    -/
    rwa [t.proj_toFun e₁ he₁, t.proj_toFun e₂ he₂]
    /-
      🎉 no goals
    -/


theorem eq_of_comp_eq [PreconnectedSpace A] (h₁ : Continuous g₁) (h₂ : Continuous g₂)
    (he : f ∘ g₁ = f ∘ g₂) (a : A) (ha : g₁ a = g₂ a) : g₁ = g₂ :=
  hf.isSeparatedMap.eq_of_comp_eq hf.isLocalHomeomorph.isLocallyInjective h₁ h₂ he a ha


theorem const_of_comp [PreconnectedSpace A] (cont : Continuous g)
    (he : ∀ a a', f (g a) = f (g a')) (a a') : g a = g a' :=
  hf.isSeparatedMap.const_of_comp hf.isLocalHomeomorph.isLocallyInjective cont he a a'


theorem eqOn_of_comp_eqOn (hs : IsPreconnected s) (h₁ : ContinuousOn g₁ s) (h₂ : ContinuousOn g₂ s)
    (he : s.EqOn (f ∘ g₁) (f ∘ g₂)) {a : A} (has : a ∈ s) (ha : g₁ a = g₂ a) : s.EqOn g₁ g₂ :=
  hf.isSeparatedMap.eqOn_of_comp_eqOn hf.isLocalHomeomorph.isLocallyInjective hs h₁ h₂ he has ha


theorem constOn_of_comp (hs : IsPreconnected s) (cont : ContinuousOn g s)
    (he : ∀ a ∈ s, ∀ a' ∈ s, f (g a) = f (g a'))
    {a a'} (ha : a ∈ s) (ha' : a' ∈ s) : g a = g a' :=
  hf.isSeparatedMap.constOn_of_comp hf.isLocalHomeomorph.isLocallyInjective hs cont he ha ha'


protected theorem IsFiberBundle.isCoveringMap {F : Type*} [TopologicalSpace F] [DiscreteTopology F]
    (hf : ∀ x : X, ∃ e : Trivialization F f, x ∈ e.baseSet) : IsCoveringMap f :=
  IsCoveringMap.mk f (fun _ => F) (fun x => Classical.choose (hf x)) fun x =>
    Classical.choose_spec (hf x)


protected theorem FiberBundle.isCoveringMap {F : Type*} {E : X → Type*} [TopologicalSpace F]
    [DiscreteTopology F] [TopologicalSpace (Bundle.TotalSpace F E)] [∀ x, TopologicalSpace (E x)]
    [FiberBundle F E] : IsCoveringMap (π F E) :=
  IsFiberBundle.isCoveringMap fun x => ⟨trivializationAt F E x, mem_baseSet_trivializationAt F E x⟩

