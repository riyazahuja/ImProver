/-- Structure recording good behavior of a property of a triple `(f, s, x)` where `f` is a function,
`s` a set and `x` a point. Good behavior here means locality and invariance under given groupoids
(both in the source and in the target). Given such a good behavior, the lift of this property
to charted spaces admitting these groupoids will inherit the good behavior. -/
structure LocalInvariantProp (P : (H → H') → Set H → H → Prop) : Prop where
  is_local : ∀ {s x u} {f : H → H'}, IsOpen u → x ∈ u → (P f s x ↔ P f (s ∩ u) x)
  right_invariance' : ∀ {s x f} {e : PartialHomeomorph H H},
    e ∈ G → x ∈ e.source → P f s x → P (f ∘ e.symm) (e.symm ⁻¹' s) (e x)
  congr_of_forall : ∀ {s x} {f g : H → H'}, (∀ y ∈ s, f y = g y) → f x = g x → P f s x → P g s x
  left_invariance' : ∀ {s x f} {e' : PartialHomeomorph H' H'},
    e' ∈ G' → s ⊆ f ⁻¹' e'.source → f x ∈ e'.source → P f s x → P (e' ∘ f) s x


theorem congr_set {s t : Set H} {x : H} {f : H → H'} (hu : s =ᶠ[𝓝 x] t) : P f s x ↔ P f t x := by
  /-
    H : Type u_1
    H' : Type u_3
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace H'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    hG : G.LocalInvariantProp G' P
    s t : Set H
    x : H
    f : H → H'
    hu : (nhds x).EventuallyEq s t
    ⊢ Iff (P f s x) (P f t x)
  -/
  obtain ⟨o, host, ho, hxo⟩ := mem_nhds_iff.mp hu.mem_iff
  /-
    case intro.intro.intro
    H : Type u_1
    H' : Type u_3
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace H'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    hG : G.LocalInvariantProp G' P
    s t : Set H
    x : H
    f : H → H'
    hu : (nhds x).EventuallyEq s t
    o : Set H
    host : HasSubset.Subset o (setOf fun x => (fun x => Iff (Membership.mem s x) ( …
    ho : IsOpen o
    hxo : Membership.mem o x
    ⊢ Iff (P f s x) (P f t x)
  -/
  simp_rw [subset_def, mem_setOf, ← and_congr_left_iff, ← mem_inter_iff, ← Set.ext_iff] at host
  /-
    case intro.intro.intro
    H : Type u_1
    H' : Type u_3
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace H'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    hG : G.LocalInvariantProp G' P
    s t : Set H
    x : H
    f : H → H'
    hu : (nhds x).EventuallyEq s t
    o : Set H
    ho : IsOpen o
    hxo : Membership.mem o x
    host : Eq (Inter.inter s o) (Inter.inter t o)
    ⊢ Iff (P f s x) (P f t x)
  -/
  rw [hG.is_local ho hxo, host, ← hG.is_local ho hxo]
  /-
    🎉 no goals
  -/


theorem is_local_nhds {s u : Set H} {x : H} {f : H → H'} (hu : u ∈ 𝓝[s] x) :
    P f s x ↔ P f (s ∩ u) x :=
  hG.congr_set <| mem_nhdsWithin_iff_eventuallyEq.mp hu


theorem congr_iff_nhdsWithin {s : Set H} {x : H} {f g : H → H'} (h1 : f =ᶠ[𝓝[s] x] g)
    (h2 : f x = g x) : P f s x ↔ P g s x := by
  /-
    H : Type u_1
    H' : Type u_3
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace H'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    hG : G.LocalInvariantProp G' P
    s : Set H
    x : H
    f g : H → H'
    h1 : (nhdsWithin x s).EventuallyEq f g
    h2 : Eq (f x) (g x)
    ⊢ Iff (P f s x) (P g s x)
  -/
  simp_rw [hG.is_local_nhds h1]
  /-
    H : Type u_1
    H' : Type u_3
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace H'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    hG : G.LocalInvariantProp G' P
    s : Set H
    x : H
    f g : H → H'
    h1 : (nhdsWithin x s).EventuallyEq f g
    h2 : Eq (f x) (g x)
    ⊢ Iff (P f (Inter.inter s (setOf fun x => Eq (f x) (g x))) x) (P g (Inter.inte …
  -/
  exact ⟨hG.congr_of_forall (fun y hy ↦ hy.2) h2, hG.congr_of_forall (fun y hy ↦ hy.2.symm) h2.symm⟩
  /-
    🎉 no goals
  -/


theorem congr_nhdsWithin {s : Set H} {x : H} {f g : H → H'} (h1 : f =ᶠ[𝓝[s] x] g) (h2 : f x = g x)
    (hP : P f s x) : P g s x :=
  (hG.congr_iff_nhdsWithin h1 h2).mp hP


theorem congr_nhdsWithin' {s : Set H} {x : H} {f g : H → H'} (h1 : f =ᶠ[𝓝[s] x] g) (h2 : f x = g x)
    (hP : P g s x) : P f s x :=
  (hG.congr_iff_nhdsWithin h1 h2).mpr hP


theorem congr_iff {s : Set H} {x : H} {f g : H → H'} (h : f =ᶠ[𝓝 x] g) : P f s x ↔ P g s x :=
  hG.congr_iff_nhdsWithin (mem_nhdsWithin_of_mem_nhds h) (mem_of_mem_nhds h : _)


theorem congr {s : Set H} {x : H} {f g : H → H'} (h : f =ᶠ[𝓝 x] g) (hP : P f s x) : P g s x :=
  (hG.congr_iff h).mp hP


theorem congr' {s : Set H} {x : H} {f g : H → H'} (h : f =ᶠ[𝓝 x] g) (hP : P g s x) : P f s x :=
  hG.congr h.symm hP


theorem left_invariance {s : Set H} {x : H} {f : H → H'} {e' : PartialHomeomorph H' H'}
    (he' : e' ∈ G') (hfs : ContinuousWithinAt f s x) (hxe' : f x ∈ e'.source) :
    P (e' ∘ f) s x ↔ P f s x := by
  /-
    H : Type u_1
    H' : Type u_3
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace H'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    hG : G.LocalInvariantProp G' P
    s : Set H
    x : H
    f : H → H'
    e' : PartialHomeomorph H' H'
    he' : Membership.mem G' e'
    hfs : ContinuousWithinAt f s x
    hxe' : Membership.mem e'.source (f x)
    ⊢ Iff (P (Function.comp (↑e') f) s x) (P f s x)
  -/
  have h2f := hfs.preimage_mem_nhdsWithin (e'.open_source.mem_nhds hxe')
  have h3f :=
    ((e'.continuousAt hxe').comp_continuousWithinAt hfs).preimage_mem_nhdsWithin <|
      e'.symm.open_source.mem_nhds <| e'.mapsTo hxe'
  /-
    H : Type u_1
    H' : Type u_3
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace H'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    hG : G.LocalInvariantProp G' P
    s : Set H
    x : H
    f : H → H'
    e' : PartialHomeomorph H' H'
    he' : Membership.mem G' e'
    hfs : ContinuousWithinAt f s x
    hxe' : Membership.mem e'.source (f x)
    h2f : Membership.mem (nhdsWithin x s) (Set.preimage f e'.source)
    h3f : Membership.mem (nhdsWithin x s) (Set.preimage (Function.comp (↑e') f) e' …
    ⊢ Iff (P (Function.comp (↑e') f) s x) (P f s x)
  -/
  constructor
    /-
      case mp
      H : Type u_1
      H' : Type u_3
      inst✝¹ : TopologicalSpace H
      inst✝ : TopologicalSpace H'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      P : (H → H') → Set H → H → Prop
      hG : G.LocalInvariantProp G' P
      s : Set H
      x : H
      f : H → H'
      e' : PartialHomeomorph H' H'
      he' : Membership.mem G' e'
      hfs : ContinuousWithinAt f s x
      hxe' : Membership.mem e'.source (f x)
      h2f : Membership.mem (nhdsWithin x s) (Set.preimage f e'.source)
      h3f : Membership.mem (nhdsWithin x s) (Set.preimage (Function.comp (↑e') f) e' …
      ⊢ P (Function.comp (↑e') f) s x → P f s x
    -/
  · intro h
    /-
      case mp
      H : Type u_1
      H' : Type u_3
      inst✝¹ : TopologicalSpace H
      inst✝ : TopologicalSpace H'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      P : (H → H') → Set H → H → Prop
      hG : G.LocalInvariantProp G' P
      s : Set H
      x : H
      f : H → H'
      e' : PartialHomeomorph H' H'
      he' : Membership.mem G' e'
      hfs : ContinuousWithinAt f s x
      hxe' : Membership.mem e'.source (f x)
      h2f : Membership.mem (nhdsWithin x s) (Set.preimage f e'.source)
      h3f : Membership.mem (nhdsWithin x s) (Set.preimage (Function.comp (↑e') f) e' …
      h : P (Function.comp (↑e') f) s x
      ⊢ P f s x
    -/
    rw [hG.is_local_nhds h3f] at h
    /-
      case mp
      H : Type u_1
      H' : Type u_3
      inst✝¹ : TopologicalSpace H
      inst✝ : TopologicalSpace H'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      P : (H → H') → Set H → H → Prop
      hG : G.LocalInvariantProp G' P
      s : Set H
      x : H
      f : H → H'
      e' : PartialHomeomorph H' H'
      he' : Membership.mem G' e'
      hfs : ContinuousWithinAt f s x
      hxe' : Membership.mem e'.source (f x)
      h2f : Membership.mem (nhdsWithin x s) (Set.preimage f e'.source)
      h3f : Membership.mem (nhdsWithin x s) (Set.preimage (Function.comp (↑e') f) e' …
      h : P (Function.comp (↑e') f) (Inter.inter s (Set.preimage (Function.comp (↑e' …
      ⊢ P f s x
    -/
    have h2 := hG.left_invariance' (G'.symm he') inter_subset_right (e'.mapsTo hxe') h
    /-
      case mp
      H : Type u_1
      H' : Type u_3
      inst✝¹ : TopologicalSpace H
      inst✝ : TopologicalSpace H'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      P : (H → H') → Set H → H → Prop
      hG : G.LocalInvariantProp G' P
      s : Set H
      x : H
      f : H → H'
      e' : PartialHomeomorph H' H'
      he' : Membership.mem G' e'
      hfs : ContinuousWithinAt f s x
      hxe' : Membership.mem e'.source (f x)
      h2f : Membership.mem (nhdsWithin x s) (Set.preimage f e'.source)
      h3f : Membership.mem (nhdsWithin x s) (Set.preimage (Function.comp (↑e') f) e' …
      h : P (Function.comp (↑e') f) (Inter.inter s (Set.preimage (Function.comp (↑e' …
      h2 : P (Function.comp (↑e'.symm) (Function.comp (↑e') f)) (Inter.inter s (Set. …
      ⊢ P f s x
    -/
    rw [← hG.is_local_nhds h3f] at h2
    /-
      case mp
      H : Type u_1
      H' : Type u_3
      inst✝¹ : TopologicalSpace H
      inst✝ : TopologicalSpace H'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      P : (H → H') → Set H → H → Prop
      hG : G.LocalInvariantProp G' P
      s : Set H
      x : H
      f : H → H'
      e' : PartialHomeomorph H' H'
      he' : Membership.mem G' e'
      hfs : ContinuousWithinAt f s x
      hxe' : Membership.mem e'.source (f x)
      h2f : Membership.mem (nhdsWithin x s) (Set.preimage f e'.source)
      h3f : Membership.mem (nhdsWithin x s) (Set.preimage (Function.comp (↑e') f) e' …
      h : P (Function.comp (↑e') f) (Inter.inter s (Set.preimage (Function.comp (↑e' …
      h2 : P (Function.comp (↑e'.symm) (Function.comp (↑e') f)) s x
      ⊢ P f s x
    -/
    refine hG.congr_nhdsWithin ?_ (e'.left_inv hxe') h2
    /-
      case mp
      H : Type u_1
      H' : Type u_3
      inst✝¹ : TopologicalSpace H
      inst✝ : TopologicalSpace H'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      P : (H → H') → Set H → H → Prop
      hG : G.LocalInvariantProp G' P
      s : Set H
      x : H
      f : H → H'
      e' : PartialHomeomorph H' H'
      he' : Membership.mem G' e'
      hfs : ContinuousWithinAt f s x
      hxe' : Membership.mem e'.source (f x)
      h2f : Membership.mem (nhdsWithin x s) (Set.preimage f e'.source)
      h3f : Membership.mem (nhdsWithin x s) (Set.preimage (Function.comp (↑e') f) e' …
      h : P (Function.comp (↑e') f) (Inter.inter s (Set.preimage (Function.comp (↑e' …
      h2 : P (Function.comp (↑e'.symm) (Function.comp (↑e') f)) s x
      ⊢ (nhdsWithin x s).EventuallyEq (Function.comp (↑e'.symm) (Function.comp (↑e') …
    -/
    exact eventually_of_mem h2f fun x' ↦ e'.left_inv
    /-
      🎉 no goals
    -/
    /-
      case mpr
      H : Type u_1
      H' : Type u_3
      inst✝¹ : TopologicalSpace H
      inst✝ : TopologicalSpace H'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      P : (H → H') → Set H → H → Prop
      hG : G.LocalInvariantProp G' P
      s : Set H
      x : H
      f : H → H'
      e' : PartialHomeomorph H' H'
      he' : Membership.mem G' e'
      hfs : ContinuousWithinAt f s x
      hxe' : Membership.mem e'.source (f x)
      h2f : Membership.mem (nhdsWithin x s) (Set.preimage f e'.source)
      h3f : Membership.mem (nhdsWithin x s) (Set.preimage (Function.comp (↑e') f) e' …
      ⊢ P f s x → P (Function.comp (↑e') f) s x
    -/
  · simp_rw [hG.is_local_nhds h2f]
    /-
      case mpr
      H : Type u_1
      H' : Type u_3
      inst✝¹ : TopologicalSpace H
      inst✝ : TopologicalSpace H'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      P : (H → H') → Set H → H → Prop
      hG : G.LocalInvariantProp G' P
      s : Set H
      x : H
      f : H → H'
      e' : PartialHomeomorph H' H'
      he' : Membership.mem G' e'
      hfs : ContinuousWithinAt f s x
      hxe' : Membership.mem e'.source (f x)
      h2f : Membership.mem (nhdsWithin x s) (Set.preimage f e'.source)
      h3f : Membership.mem (nhdsWithin x s) (Set.preimage (Function.comp (↑e') f) e' …
      ⊢ P f (Inter.inter s (Set.preimage f e'.source)) x → P (Function.comp (↑e') f) …
    -/
    exact hG.left_invariance' he' inter_subset_right hxe'
    /-
      🎉 no goals
    -/


theorem right_invariance {s : Set H} {x : H} {f : H → H'} {e : PartialHomeomorph H H} (he : e ∈ G)
    (hxe : x ∈ e.source) : P (f ∘ e.symm) (e.symm ⁻¹' s) (e x) ↔ P f s x := by
  /-
    H : Type u_1
    H' : Type u_3
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace H'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    hG : G.LocalInvariantProp G' P
    s : Set H
    x : H
    f : H → H'
    e : PartialHomeomorph H H
    he : Membership.mem G e
    hxe : Membership.mem e.source x
    ⊢ Iff (P (Function.comp f ↑e.symm) (Set.preimage (↑e.symm) s) (↑e x)) (P f s x)
  -/
  refine ⟨fun h ↦ ?_, hG.right_invariance' he hxe⟩
  /-
    H : Type u_1
    H' : Type u_3
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace H'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    hG : G.LocalInvariantProp G' P
    s : Set H
    x : H
    f : H → H'
    e : PartialHomeomorph H H
    he : Membership.mem G e
    hxe : Membership.mem e.source x
    h : P (Function.comp f ↑e.symm) (Set.preimage (↑e.symm) s) (↑e x)
    ⊢ P f s x
  -/
  have := hG.right_invariance' (G.symm he) (e.mapsTo hxe) h
  /-
    H : Type u_1
    H' : Type u_3
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace H'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    hG : G.LocalInvariantProp G' P
    s : Set H
    x : H
    f : H → H'
    e : PartialHomeomorph H H
    he : Membership.mem G e
    hxe : Membership.mem e.source x
    h : P (Function.comp f ↑e.symm) (Set.preimage (↑e.symm) s) (↑e x)
    this : P (Function.comp (Function.comp f ↑e.symm) ↑e.symm.symm) (Set.preimage  …
    ⊢ P f s x
  -/
  rw [e.symm_symm, e.left_inv hxe] at this
  /-
    H : Type u_1
    H' : Type u_3
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace H'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    hG : G.LocalInvariantProp G' P
    s : Set H
    x : H
    f : H → H'
    e : PartialHomeomorph H H
    he : Membership.mem G e
    hxe : Membership.mem e.source x
    h : P (Function.comp f ↑e.symm) (Set.preimage (↑e.symm) s) (↑e x)
    this : P (Function.comp (Function.comp f ↑e.symm) ↑e) (Set.preimage (↑e) (Set. …
    ⊢ P f s x
  -/
  refine hG.congr ?_ ((hG.congr_set ?_).mp this)
    /-
      case refine_1
      H : Type u_1
      H' : Type u_3
      inst✝¹ : TopologicalSpace H
      inst✝ : TopologicalSpace H'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      P : (H → H') → Set H → H → Prop
      hG : G.LocalInvariantProp G' P
      s : Set H
      x : H
      f : H → H'
      e : PartialHomeomorph H H
      he : Membership.mem G e
      hxe : Membership.mem e.source x
      h : P (Function.comp f ↑e.symm) (Set.preimage (↑e.symm) s) (↑e x)
      this : P (Function.comp (Function.comp f ↑e.symm) ↑e) (Set.preimage (↑e) (Set. …
      ⊢ (nhds x).EventuallyEq (Function.comp (Function.comp f ↑e.symm) ↑e) f
    -/
  · refine eventually_of_mem (e.open_source.mem_nhds hxe) fun x' hx' ↦ ?_
    /-
      case refine_1
      H : Type u_1
      H' : Type u_3
      inst✝¹ : TopologicalSpace H
      inst✝ : TopologicalSpace H'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      P : (H → H') → Set H → H → Prop
      hG : G.LocalInvariantProp G' P
      s : Set H
      x : H
      f : H → H'
      e : PartialHomeomorph H H
      he : Membership.mem G e
      hxe : Membership.mem e.source x
      h : P (Function.comp f ↑e.symm) (Set.preimage (↑e.symm) s) (↑e x)
      this : P (Function.comp (Function.comp f ↑e.symm) ↑e) (Set.preimage (↑e) (Set. …
      x' : H
      hx' : Membership.mem e.source x'
      ⊢ Eq (Function.comp (Function.comp f ↑e.symm) (↑e) x') (f x')
    -/
    simp_rw [Function.comp_apply, e.left_inv hx']
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      H : Type u_1
      H' : Type u_3
      inst✝¹ : TopologicalSpace H
      inst✝ : TopologicalSpace H'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      P : (H → H') → Set H → H → Prop
      hG : G.LocalInvariantProp G' P
      s : Set H
      x : H
      f : H → H'
      e : PartialHomeomorph H H
      he : Membership.mem G e
      hxe : Membership.mem e.source x
      h : P (Function.comp f ↑e.symm) (Set.preimage (↑e.symm) s) (↑e x)
      this : P (Function.comp (Function.comp f ↑e.symm) ↑e) (Set.preimage (↑e) (Set. …
      ⊢ (nhds x).EventuallyEq (Set.preimage (↑e) (Set.preimage (↑e.symm) s)) s
    -/
  · rw [eventuallyEq_set]
    /-
      case refine_2
      H : Type u_1
      H' : Type u_3
      inst✝¹ : TopologicalSpace H
      inst✝ : TopologicalSpace H'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      P : (H → H') → Set H → H → Prop
      hG : G.LocalInvariantProp G' P
      s : Set H
      x : H
      f : H → H'
      e : PartialHomeomorph H H
      he : Membership.mem G e
      hxe : Membership.mem e.source x
      h : P (Function.comp f ↑e.symm) (Set.preimage (↑e.symm) s) (↑e x)
      this : P (Function.comp (Function.comp f ↑e.symm) ↑e) (Set.preimage (↑e) (Set. …
      ⊢ Filter.Eventually (fun x => Iff (Membership.mem (Set.preimage (↑e) (Set.prei …
    -/
    refine eventually_of_mem (e.open_source.mem_nhds hxe) fun x' hx' ↦ ?_
    /-
      case refine_2
      H : Type u_1
      H' : Type u_3
      inst✝¹ : TopologicalSpace H
      inst✝ : TopologicalSpace H'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      P : (H → H') → Set H → H → Prop
      hG : G.LocalInvariantProp G' P
      s : Set H
      x : H
      f : H → H'
      e : PartialHomeomorph H H
      he : Membership.mem G e
      hxe : Membership.mem e.source x
      h : P (Function.comp f ↑e.symm) (Set.preimage (↑e.symm) s) (↑e x)
      this : P (Function.comp (Function.comp f ↑e.symm) ↑e) (Set.preimage (↑e) (Set. …
      x' : H
      hx' : Membership.mem e.source x'
      ⊢ Iff (Membership.mem (Set.preimage (↑e) (Set.preimage (↑e.symm) s)) x') (Memb …
    -/
    simp_rw [mem_preimage, e.left_inv hx']
    /-
      🎉 no goals
    -/


/-- Given a property of germs of functions and sets in the model space, then one defines
a corresponding property in a charted space, by requiring that it holds at the preferred chart at
this point. (When the property is local and invariant, it will in fact hold using any chart, see
`liftPropWithinAt_indep_chart`). We require continuity in the lifted property, as otherwise one
single chart might fail to capture the behavior of the function.
-/
@[mk_iff liftPropWithinAt_iff']
structure LiftPropWithinAt (P : (H → H') → Set H → H → Prop) (f : M → M') (s : Set M) (x : M) :
    Prop where
  continuousWithinAt : ContinuousWithinAt f s x
  prop : P (chartAt H' (f x) ∘ f ∘ (chartAt H x).symm) ((chartAt H x).symm ⁻¹' s) (chartAt H x x)


/-- Given a property of germs of functions and sets in the model space, then one defines
a corresponding property of functions on sets in a charted space, by requiring that it holds
around each point of the set, in the preferred charts. -/
def LiftPropOn (P : (H → H') → Set H → H → Prop) (f : M → M') (s : Set M) :=
  ∀ x ∈ s, LiftPropWithinAt P f s x


/-- Given a property of germs of functions and sets in the model space, then one defines
a corresponding property of a function at a point in a charted space, by requiring that it holds
in the preferred chart. -/
def LiftPropAt (P : (H → H') → Set H → H → Prop) (f : M → M') (x : M) :=
  LiftPropWithinAt P f univ x


theorem liftPropAt_iff {P : (H → H') → Set H → H → Prop} {f : M → M'} {x : M} :
    LiftPropAt P f x ↔
      ContinuousAt f x ∧ P (chartAt H' (f x) ∘ f ∘ (chartAt H x).symm) univ (chartAt H x x) := by
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    P : (H → H') → Set H → H → Prop
    f : M → M'
    x : M
    ⊢ Iff (ChartedSpace.LiftPropAt P f x) (And (ContinuousAt f x) (P (Function.com …
  -/
  rw [LiftPropAt, liftPropWithinAt_iff', continuousWithinAt_univ, preimage_univ]
  /-
    🎉 no goals
  -/


/-- Given a property of germs of functions and sets in the model space, then one defines
a corresponding property of a function in a charted space, by requiring that it holds
in the preferred chart around every point. -/
def LiftProp (P : (H → H') → Set H → H → Prop) (f : M → M') :=
  ∀ x, LiftPropAt P f x


theorem liftProp_iff {P : (H → H') → Set H → H → Prop} {f : M → M'} :
    LiftProp P f ↔
      Continuous f ∧ ∀ x, P (chartAt H' (f x) ∘ f ∘ (chartAt H x).symm) univ (chartAt H x x) := by
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    P : (H → H') → Set H → H → Prop
    f : M → M'
    ⊢ Iff (ChartedSpace.LiftProp P f) (And (Continuous f) (∀ (x : M), P (Function. …
  -/
  simp_rw [LiftProp, liftPropAt_iff, forall_and, continuous_iff_continuousAt]
  /-
    🎉 no goals
  -/


theorem liftPropWithinAt_univ : LiftPropWithinAt P g univ x ↔ LiftPropAt P g x := Iff.rfl


theorem liftPropOn_univ : LiftPropOn P g univ ↔ LiftProp P g := by
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    ⊢ Iff (ChartedSpace.LiftPropOn P g Set.univ) (ChartedSpace.LiftProp P g)
  -/
  simp [LiftPropOn, LiftProp, LiftPropAt]
  /-
    🎉 no goals
  -/


theorem liftPropWithinAt_self {f : H → H'} {s : Set H} {x : H} :
    LiftPropWithinAt P f s x ↔ ContinuousWithinAt f s x ∧ P f s x :=
  liftPropWithinAt_iff' ..


theorem liftPropWithinAt_self_source {f : H → M'} {s : Set H} {x : H} :
    LiftPropWithinAt P f s x ↔ ContinuousWithinAt f s x ∧ P (chartAt H' (f x) ∘ f) s x :=
  liftPropWithinAt_iff' ..


theorem liftPropWithinAt_self_target {f : M → H'} :
    LiftPropWithinAt P f s x ↔ ContinuousWithinAt f s x ∧
      P (f ∘ (chartAt H x).symm) ((chartAt H x).symm ⁻¹' s) (chartAt H x x) :=
  liftPropWithinAt_iff' ..


/-- `LiftPropWithinAt P f s x` is equivalent to a definition where we restrict the set we are
  considering to the domain of the charts at `x` and `f x`. -/
theorem liftPropWithinAt_iff {f : M → M'} :
    LiftPropWithinAt P f s x ↔
      ContinuousWithinAt f s x ∧
        P (chartAt H' (f x) ∘ f ∘ (chartAt H x).symm)
          ((chartAt H x).target ∩ (chartAt H x).symm ⁻¹' (s ∩ f ⁻¹' (chartAt H' (f x)).source))
          (chartAt H x x) := by
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    s : Set M
    x : M
    hG : G.LocalInvariantProp G' P
    f : M → M'
    ⊢ Iff (ChartedSpace.LiftPropWithinAt P f s x) (And (ContinuousWithinAt f s x)  …
  -/
  rw [liftPropWithinAt_iff']
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    s : Set M
    x : M
    hG : G.LocalInvariantProp G' P
    f : M → M'
    ⊢ Iff (And (ContinuousWithinAt f s x) (P (Function.comp (↑(chartAt H' (f x)))  …
  -/
  refine and_congr_right fun hf ↦ hG.congr_set ?_
  exact PartialHomeomorph.preimage_eventuallyEq_target_inter_preimage_inter hf
    (mem_chart_source H x) (chart_source_mem_nhds H' (f x))


theorem liftPropWithinAt_indep_chart_source_aux (g : M → H') (he : e ∈ G.maximalAtlas M)
    (xe : x ∈ e.source) (he' : e' ∈ G.maximalAtlas M) (xe' : x ∈ e'.source) :
    P (g ∘ e.symm) (e.symm ⁻¹' s) (e x) ↔ P (g ∘ e'.symm) (e'.symm ⁻¹' s) (e' x) := by
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : TopologicalSpace H'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    e e' : PartialHomeomorph M H
    P : (H → H') → Set H → H → Prop
    s : Set M
    x : M
    hG : G.LocalInvariantProp G' P
    g : M → H'
    he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
    xe : Membership.mem e.source x
    he' : Membership.mem (StructureGroupoid.maximalAtlas M G) e'
    xe' : Membership.mem e'.source x
    ⊢ Iff (P (Function.comp g ↑e.symm) (Set.preimage (↑e.symm) s) (↑e x)) (P (Func …
  -/
  rw [← hG.right_invariance (compatible_of_mem_maximalAtlas he he')]
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : TopologicalSpace H'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    e e' : PartialHomeomorph M H
    P : (H → H') → Set H → H → Prop
    s : Set M
    x : M
    hG : G.LocalInvariantProp G' P
    g : M → H'
    he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
    xe : Membership.mem e.source x
    he' : Membership.mem (StructureGroupoid.maximalAtlas M G) e'
    xe' : Membership.mem e'.source x
    ⊢ Iff (P (Function.comp (Function.comp g ↑e.symm) ↑(e.symm.trans e').symm) (Se …
  -/
  swap; · simp only [xe, xe', mfld_simps]
          /-
            🎉 no goals
          -/
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : TopologicalSpace H'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    e e' : PartialHomeomorph M H
    P : (H → H') → Set H → H → Prop
    s : Set M
    x : M
    hG : G.LocalInvariantProp G' P
    g : M → H'
    he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
    xe : Membership.mem e.source x
    he' : Membership.mem (StructureGroupoid.maximalAtlas M G) e'
    xe' : Membership.mem e'.source x
    ⊢ Iff (P (Function.comp (Function.comp g ↑e.symm) ↑(e.symm.trans e').symm) (Se …
  -/
  simp_rw [PartialHomeomorph.trans_apply, e.left_inv xe]
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : TopologicalSpace H'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    e e' : PartialHomeomorph M H
    P : (H → H') → Set H → H → Prop
    s : Set M
    x : M
    hG : G.LocalInvariantProp G' P
    g : M → H'
    he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
    xe : Membership.mem e.source x
    he' : Membership.mem (StructureGroupoid.maximalAtlas M G) e'
    xe' : Membership.mem e'.source x
    ⊢ Iff (P (Function.comp (Function.comp g ↑e.symm) ↑(e.symm.trans e').symm) (Se …
  -/
  rw [hG.congr_iff]
    /-
      H : Type u_1
      M : Type u_2
      H' : Type u_3
      inst✝³ : TopologicalSpace H
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      inst✝ : TopologicalSpace H'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      e e' : PartialHomeomorph M H
      P : (H → H') → Set H → H → Prop
      s : Set M
      x : M
      hG : G.LocalInvariantProp G' P
      g : M → H'
      he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
      xe : Membership.mem e.source x
      he' : Membership.mem (StructureGroupoid.maximalAtlas M G) e'
      xe' : Membership.mem e'.source x
      ⊢ Iff (P ?m.21415 (Set.preimage (↑(e.symm.trans e').symm) (Set.preimage (↑e.sy …
    -/
  · refine hG.congr_set ?_
    /-
      H : Type u_1
      M : Type u_2
      H' : Type u_3
      inst✝³ : TopologicalSpace H
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      inst✝ : TopologicalSpace H'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      e e' : PartialHomeomorph M H
      P : (H → H') → Set H → H → Prop
      s : Set M
      x : M
      hG : G.LocalInvariantProp G' P
      g : M → H'
      he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
      xe : Membership.mem e.source x
      he' : Membership.mem (StructureGroupoid.maximalAtlas M G) e'
      xe' : Membership.mem e'.source x
      ⊢ (nhds (↑e' x)).EventuallyEq (Set.preimage (↑(e.symm.trans e').symm) (Set.pre …
    -/
    refine (eventually_of_mem ?_ fun y (hy : y ∈ e'.symm ⁻¹' e.source) ↦ ?_).set_eq
      /-
        case refine_1
        H : Type u_1
        M : Type u_2
        H' : Type u_3
        inst✝³ : TopologicalSpace H
        inst✝² : TopologicalSpace M
        inst✝¹ : ChartedSpace H M
        inst✝ : TopologicalSpace H'
        G : StructureGroupoid H
        G' : StructureGroupoid H'
        e e' : PartialHomeomorph M H
        P : (H → H') → Set H → H → Prop
        s : Set M
        x : M
        hG : G.LocalInvariantProp G' P
        g : M → H'
        he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
        xe : Membership.mem e.source x
        he' : Membership.mem (StructureGroupoid.maximalAtlas M G) e'
        xe' : Membership.mem e'.source x
        ⊢ Membership.mem (nhds (↑e' x)) (Set.preimage (↑e'.symm) e.source)
      -/
    · refine (e'.symm.continuousAt <| e'.mapsTo xe').preimage_mem_nhds (e.open_source.mem_nhds ?_)
      /-
        case refine_1
        H : Type u_1
        M : Type u_2
        H' : Type u_3
        inst✝³ : TopologicalSpace H
        inst✝² : TopologicalSpace M
        inst✝¹ : ChartedSpace H M
        inst✝ : TopologicalSpace H'
        G : StructureGroupoid H
        G' : StructureGroupoid H'
        e e' : PartialHomeomorph M H
        P : (H → H') → Set H → H → Prop
        s : Set M
        x : M
        hG : G.LocalInvariantProp G' P
        g : M → H'
        he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
        xe : Membership.mem e.source x
        he' : Membership.mem (StructureGroupoid.maximalAtlas M G) e'
        xe' : Membership.mem e'.source x
        ⊢ Membership.mem e.source (↑e'.symm (↑e' x))
      -/
      simp_rw [e'.left_inv xe', xe]
      /-
        🎉 no goals
      -/
    simp_rw [mem_preimage, PartialHomeomorph.coe_trans_symm, PartialHomeomorph.symm_symm,
      Function.comp_apply, e.left_inv hy]
    /-
      H : Type u_1
      M : Type u_2
      H' : Type u_3
      inst✝³ : TopologicalSpace H
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      inst✝ : TopologicalSpace H'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      e e' : PartialHomeomorph M H
      P : (H → H') → Set H → H → Prop
      s : Set M
      x : M
      hG : G.LocalInvariantProp G' P
      g : M → H'
      he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
      xe : Membership.mem e.source x
      he' : Membership.mem (StructureGroupoid.maximalAtlas M G) e'
      xe' : Membership.mem e'.source x
      ⊢ (nhds (↑e' x)).EventuallyEq (Function.comp (Function.comp g ↑e.symm) ↑(e.sym …
    -/
  · refine ((e'.eventually_nhds' _ xe').mpr <| e.eventually_left_inverse xe).mono fun y hy ↦ ?_
    /-
      H : Type u_1
      M : Type u_2
      H' : Type u_3
      inst✝³ : TopologicalSpace H
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      inst✝ : TopologicalSpace H'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      e e' : PartialHomeomorph M H
      P : (H → H') → Set H → H → Prop
      s : Set M
      x : M
      hG : G.LocalInvariantProp G' P
      g : M → H'
      he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
      xe : Membership.mem e.source x
      he' : Membership.mem (StructureGroupoid.maximalAtlas M G) e'
      xe' : Membership.mem e'.source x
      y : H
      hy : Eq (↑e.symm (↑e (↑e'.symm y))) (↑e'.symm y)
      ⊢ Eq (Function.comp (Function.comp g ↑e.symm) (↑(e.symm.trans e').symm) y) (Fu …
    -/
    simp only [mfld_simps]
    /-
      H : Type u_1
      M : Type u_2
      H' : Type u_3
      inst✝³ : TopologicalSpace H
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      inst✝ : TopologicalSpace H'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      e e' : PartialHomeomorph M H
      P : (H → H') → Set H → H → Prop
      s : Set M
      x : M
      hG : G.LocalInvariantProp G' P
      g : M → H'
      he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
      xe : Membership.mem e.source x
      he' : Membership.mem (StructureGroupoid.maximalAtlas M G) e'
      xe' : Membership.mem e'.source x
      y : H
      hy : Eq (↑e.symm (↑e (↑e'.symm y))) (↑e'.symm y)
      ⊢ Eq (g (↑e.symm (↑e (↑e'.symm y)))) (g (↑e'.symm y))
    -/
    rw [hy]
    /-
      🎉 no goals
    -/


theorem liftPropWithinAt_indep_chart_target_aux2 (g : H → M') {x : H} {s : Set H}
    (hf : f ∈ G'.maximalAtlas M') (xf : g x ∈ f.source) (hf' : f' ∈ G'.maximalAtlas M')
    (xf' : g x ∈ f'.source) (hgs : ContinuousWithinAt g s x) : P (f ∘ g) s x ↔ P (f' ∘ g) s x := by
  /-
    H : Type u_1
    H' : Type u_3
    M' : Type u_4
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    f f' : PartialHomeomorph M' H'
    P : (H → H') → Set H → H → Prop
    hG : G.LocalInvariantProp G' P
    g : H → M'
    x : H
    s : Set H
    hf : Membership.mem (StructureGroupoid.maximalAtlas M' G') f
    xf : Membership.mem f.source (g x)
    hf' : Membership.mem (StructureGroupoid.maximalAtlas M' G') f'
    xf' : Membership.mem f'.source (g x)
    hgs : ContinuousWithinAt g s x
    ⊢ Iff (P (Function.comp (↑f) g) s x) (P (Function.comp (↑f') g) s x)
  -/
  have hcont : ContinuousWithinAt (f ∘ g) s x := (f.continuousAt xf).comp_continuousWithinAt hgs
  rw [← hG.left_invariance (compatible_of_mem_maximalAtlas hf hf') hcont
      (by simp only [xf, xf', mfld_simps])]
  /-
    H : Type u_1
    H' : Type u_3
    M' : Type u_4
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    f f' : PartialHomeomorph M' H'
    P : (H → H') → Set H → H → Prop
    hG : G.LocalInvariantProp G' P
    g : H → M'
    x : H
    s : Set H
    hf : Membership.mem (StructureGroupoid.maximalAtlas M' G') f
    xf : Membership.mem f.source (g x)
    hf' : Membership.mem (StructureGroupoid.maximalAtlas M' G') f'
    xf' : Membership.mem f'.source (g x)
    hgs : ContinuousWithinAt g s x
    hcont : ContinuousWithinAt (Function.comp (↑f) g) s x
    ⊢ Iff (P (Function.comp (↑(f.symm.trans f')) (Function.comp (↑f) g)) s x) (P ( …
  -/
  refine hG.congr_iff_nhdsWithin ?_ (by simp only [xf, mfld_simps])
  /-
    H : Type u_1
    H' : Type u_3
    M' : Type u_4
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    f f' : PartialHomeomorph M' H'
    P : (H → H') → Set H → H → Prop
    hG : G.LocalInvariantProp G' P
    g : H → M'
    x : H
    s : Set H
    hf : Membership.mem (StructureGroupoid.maximalAtlas M' G') f
    xf : Membership.mem f.source (g x)
    hf' : Membership.mem (StructureGroupoid.maximalAtlas M' G') f'
    xf' : Membership.mem f'.source (g x)
    hgs : ContinuousWithinAt g s x
    hcont : ContinuousWithinAt (Function.comp (↑f) g) s x
    ⊢ (nhdsWithin x s).EventuallyEq (Function.comp (↑(f.symm.trans f')) (Function. …
  -/
  exact (hgs.eventually <| f.eventually_left_inverse xf).mono fun y ↦ congr_arg f'
  /-
    🎉 no goals
  -/


theorem liftPropWithinAt_indep_chart_target_aux {g : X → M'} {e : PartialHomeomorph X H} {x : X}
    {s : Set X} (xe : x ∈ e.source) (hf : f ∈ G'.maximalAtlas M') (xf : g x ∈ f.source)
    (hf' : f' ∈ G'.maximalAtlas M') (xf' : g x ∈ f'.source) (hgs : ContinuousWithinAt g s x) :
    P (f ∘ g ∘ e.symm) (e.symm ⁻¹' s) (e x) ↔ P (f' ∘ g ∘ e.symm) (e.symm ⁻¹' s) (e x) := by
  /-
    H : Type u_1
    H' : Type u_3
    M' : Type u_4
    X : Type u_5
    inst✝⁴ : TopologicalSpace H
    inst✝³ : TopologicalSpace H'
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    inst✝ : TopologicalSpace X
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    f f' : PartialHomeomorph M' H'
    P : (H → H') → Set H → H → Prop
    hG : G.LocalInvariantProp G' P
    g : X → M'
    e : PartialHomeomorph X H
    x : X
    s : Set X
    xe : Membership.mem e.source x
    hf : Membership.mem (StructureGroupoid.maximalAtlas M' G') f
    xf : Membership.mem f.source (g x)
    hf' : Membership.mem (StructureGroupoid.maximalAtlas M' G') f'
    xf' : Membership.mem f'.source (g x)
    hgs : ContinuousWithinAt g s x
    ⊢ Iff (P (Function.comp (↑f) (Function.comp g ↑e.symm)) (Set.preimage (↑e.symm …
  -/
  rw [← e.left_inv xe] at xf xf' hgs
  /-
    H : Type u_1
    H' : Type u_3
    M' : Type u_4
    X : Type u_5
    inst✝⁴ : TopologicalSpace H
    inst✝³ : TopologicalSpace H'
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    inst✝ : TopologicalSpace X
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    f f' : PartialHomeomorph M' H'
    P : (H → H') → Set H → H → Prop
    hG : G.LocalInvariantProp G' P
    g : X → M'
    e : PartialHomeomorph X H
    x : X
    s : Set X
    xe : Membership.mem e.source x
    hf : Membership.mem (StructureGroupoid.maximalAtlas M' G') f
    xf : Membership.mem f.source (g (↑e.symm (↑e x)))
    hf' : Membership.mem (StructureGroupoid.maximalAtlas M' G') f'
    xf' : Membership.mem f'.source (g (↑e.symm (↑e x)))
    hgs : ContinuousWithinAt g s (↑e.symm (↑e x))
    ⊢ Iff (P (Function.comp (↑f) (Function.comp g ↑e.symm)) (Set.preimage (↑e.symm …
  -/
  refine hG.liftPropWithinAt_indep_chart_target_aux2 (g ∘ e.symm) hf xf hf' xf' ?_
  /-
    H : Type u_1
    H' : Type u_3
    M' : Type u_4
    X : Type u_5
    inst✝⁴ : TopologicalSpace H
    inst✝³ : TopologicalSpace H'
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    inst✝ : TopologicalSpace X
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    f f' : PartialHomeomorph M' H'
    P : (H → H') → Set H → H → Prop
    hG : G.LocalInvariantProp G' P
    g : X → M'
    e : PartialHomeomorph X H
    x : X
    s : Set X
    xe : Membership.mem e.source x
    hf : Membership.mem (StructureGroupoid.maximalAtlas M' G') f
    xf : Membership.mem f.source (g (↑e.symm (↑e x)))
    hf' : Membership.mem (StructureGroupoid.maximalAtlas M' G') f'
    xf' : Membership.mem f'.source (g (↑e.symm (↑e x)))
    hgs : ContinuousWithinAt g s (↑e.symm (↑e x))
    ⊢ ContinuousWithinAt (Function.comp g ↑e.symm) (Set.preimage (↑e.symm) s) (↑e x)
  -/
  exact hgs.comp (e.symm.continuousAt <| e.mapsTo xe).continuousWithinAt Subset.rfl
  /-
    🎉 no goals
  -/


/-- If a property of a germ of function `g` on a pointed set `(s, x)` is invariant under the
structure groupoid (by composition in the source space and in the target space), then
expressing it in charted spaces does not depend on the element of the maximal atlas one uses
both in the source and in the target manifolds, provided they are defined around `x` and `g x`
respectively, and provided `g` is continuous within `s` at `x` (otherwise, the local behavior
of `g` at `x` can not be captured with a chart in the target). -/
theorem liftPropWithinAt_indep_chart_aux (he : e ∈ G.maximalAtlas M) (xe : x ∈ e.source)
    (he' : e' ∈ G.maximalAtlas M) (xe' : x ∈ e'.source) (hf : f ∈ G'.maximalAtlas M')
    (xf : g x ∈ f.source) (hf' : f' ∈ G'.maximalAtlas M') (xf' : g x ∈ f'.source)
    (hgs : ContinuousWithinAt g s x) :
    P (f ∘ g ∘ e.symm) (e.symm ⁻¹' s) (e x) ↔ P (f' ∘ g ∘ e'.symm) (e'.symm ⁻¹' s) (e' x) := by
  rw [← Function.comp_assoc, hG.liftPropWithinAt_indep_chart_source_aux (f ∘ g) he xe he' xe',
    Function.comp_assoc, hG.liftPropWithinAt_indep_chart_target_aux xe' hf xf hf' xf' hgs]


theorem liftPropWithinAt_indep_chart [HasGroupoid M G] [HasGroupoid M' G']
    (he : e ∈ G.maximalAtlas M) (xe : x ∈ e.source) (hf : f ∈ G'.maximalAtlas M')
    (xf : g x ∈ f.source) :
    LiftPropWithinAt P g s x ↔
    ContinuousWithinAt g s x ∧ P (f ∘ g ∘ e.symm) (e.symm ⁻¹' s) (e x) := by
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁷ : TopologicalSpace H
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    inst✝⁴ : TopologicalSpace H'
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    e : PartialHomeomorph M H
    f : PartialHomeomorph M' H'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s : Set M
    x : M
    hG : G.LocalInvariantProp G' P
    inst✝¹ : HasGroupoid M G
    inst✝ : HasGroupoid M' G'
    he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
    xe : Membership.mem e.source x
    hf : Membership.mem (StructureGroupoid.maximalAtlas M' G') f
    xf : Membership.mem f.source (g x)
    ⊢ Iff (ChartedSpace.LiftPropWithinAt P g s x) (And (ContinuousWithinAt g s x)  …
  -/
  simp only [liftPropWithinAt_iff']
  exact and_congr_right <|
    hG.liftPropWithinAt_indep_chart_aux (chart_mem_maximalAtlas _ _) (mem_chart_source _ _) he xe
      (chart_mem_maximalAtlas _ _) (mem_chart_source _ _) hf xf


/-- A version of `liftPropWithinAt_indep_chart`, only for the source. -/
theorem liftPropWithinAt_indep_chart_source [HasGroupoid M G] (he : e ∈ G.maximalAtlas M)
    (xe : x ∈ e.source) :
    LiftPropWithinAt P g s x ↔ LiftPropWithinAt P (g ∘ e.symm) (e.symm ⁻¹' s) (e x) := by
  rw [liftPropWithinAt_self_source, liftPropWithinAt_iff',
    e.symm.continuousWithinAt_iff_continuousWithinAt_comp_right xe, e.symm_symm]
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁶ : TopologicalSpace H
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : TopologicalSpace H'
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    e : PartialHomeomorph M H
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s : Set M
    x : M
    hG : G.LocalInvariantProp G' P
    inst✝ : HasGroupoid M G
    he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
    xe : Membership.mem e.source x
    ⊢ Iff (And (ContinuousWithinAt (Function.comp g ↑e.symm) (Set.preimage (↑e.sym …
  -/
  refine and_congr Iff.rfl ?_
  rw [Function.comp_apply, e.left_inv xe, ← Function.comp_assoc,
    hG.liftPropWithinAt_indep_chart_source_aux (chartAt _ (g x) ∘ g) (chart_mem_maximalAtlas G x)
      (mem_chart_source _ x) he xe, Function.comp_assoc]


/-- A version of `liftPropWithinAt_indep_chart`, only for the target. -/
theorem liftPropWithinAt_indep_chart_target [HasGroupoid M' G'] (hf : f ∈ G'.maximalAtlas M')
    (xf : g x ∈ f.source) :
    LiftPropWithinAt P g s x ↔ ContinuousWithinAt g s x ∧ LiftPropWithinAt P (f ∘ g) s x := by
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁶ : TopologicalSpace H
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : TopologicalSpace H'
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    f : PartialHomeomorph M' H'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s : Set M
    x : M
    hG : G.LocalInvariantProp G' P
    inst✝ : HasGroupoid M' G'
    hf : Membership.mem (StructureGroupoid.maximalAtlas M' G') f
    xf : Membership.mem f.source (g x)
    ⊢ Iff (ChartedSpace.LiftPropWithinAt P g s x) (And (ContinuousWithinAt g s x)  …
  -/
  rw [liftPropWithinAt_self_target, liftPropWithinAt_iff', and_congr_right_iff]
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁶ : TopologicalSpace H
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : TopologicalSpace H'
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    f : PartialHomeomorph M' H'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s : Set M
    x : M
    hG : G.LocalInvariantProp G' P
    inst✝ : HasGroupoid M' G'
    hf : Membership.mem (StructureGroupoid.maximalAtlas M' G') f
    xf : Membership.mem f.source (g x)
    ⊢ ContinuousWithinAt g s x → Iff (P (Function.comp (↑(chartAt H' (g x))) (Func …
  -/
  intro hg
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁶ : TopologicalSpace H
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : TopologicalSpace H'
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    f : PartialHomeomorph M' H'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s : Set M
    x : M
    hG : G.LocalInvariantProp G' P
    inst✝ : HasGroupoid M' G'
    hf : Membership.mem (StructureGroupoid.maximalAtlas M' G') f
    xf : Membership.mem f.source (g x)
    hg : ContinuousWithinAt g s x
    ⊢ Iff (P (Function.comp (↑(chartAt H' (g x))) (Function.comp g ↑(chartAt H x). …
  -/
  simp_rw [(f.continuousAt xf).comp_continuousWithinAt hg, true_and]
  exact hG.liftPropWithinAt_indep_chart_target_aux (mem_chart_source _ _)
    (chart_mem_maximalAtlas _ _) (mem_chart_source _ _) hf xf hg


/-- A version of `liftPropWithinAt_indep_chart`, that uses `LiftPropWithinAt` on both sides. -/
theorem liftPropWithinAt_indep_chart' [HasGroupoid M G] [HasGroupoid M' G']
    (he : e ∈ G.maximalAtlas M) (xe : x ∈ e.source) (hf : f ∈ G'.maximalAtlas M')
    (xf : g x ∈ f.source) :
    LiftPropWithinAt P g s x ↔
      ContinuousWithinAt g s x ∧ LiftPropWithinAt P (f ∘ g ∘ e.symm) (e.symm ⁻¹' s) (e x) := by
  rw [hG.liftPropWithinAt_indep_chart he xe hf xf, liftPropWithinAt_self, and_left_comm,
    Iff.comm, and_iff_right_iff_imp]
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁷ : TopologicalSpace H
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    inst✝⁴ : TopologicalSpace H'
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    e : PartialHomeomorph M H
    f : PartialHomeomorph M' H'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s : Set M
    x : M
    hG : G.LocalInvariantProp G' P
    inst✝¹ : HasGroupoid M G
    inst✝ : HasGroupoid M' G'
    he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
    xe : Membership.mem e.source x
    hf : Membership.mem (StructureGroupoid.maximalAtlas M' G') f
    xf : Membership.mem f.source (g x)
    ⊢ And (ContinuousWithinAt g s x) (P (Function.comp (↑f) (Function.comp g ↑e.sy …
  -/
  intro h
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁷ : TopologicalSpace H
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    inst✝⁴ : TopologicalSpace H'
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    e : PartialHomeomorph M H
    f : PartialHomeomorph M' H'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s : Set M
    x : M
    hG : G.LocalInvariantProp G' P
    inst✝¹ : HasGroupoid M G
    inst✝ : HasGroupoid M' G'
    he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
    xe : Membership.mem e.source x
    hf : Membership.mem (StructureGroupoid.maximalAtlas M' G') f
    xf : Membership.mem f.source (g x)
    h : And (ContinuousWithinAt g s x) (P (Function.comp (↑f) (Function.comp g ↑e. …
    ⊢ ContinuousWithinAt (Function.comp (↑f) (Function.comp g ↑e.symm)) (Set.preim …
  -/
  have h1 := (e.symm.continuousWithinAt_iff_continuousWithinAt_comp_right xe).mp h.1
  have : ContinuousAt f ((g ∘ e.symm) (e x)) := by
    simp_rw [Function.comp, e.left_inv xe, f.continuousAt xf]
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁷ : TopologicalSpace H
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    inst✝⁴ : TopologicalSpace H'
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    e : PartialHomeomorph M H
    f : PartialHomeomorph M' H'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s : Set M
    x : M
    hG : G.LocalInvariantProp G' P
    inst✝¹ : HasGroupoid M G
    inst✝ : HasGroupoid M' G'
    he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
    xe : Membership.mem e.source x
    hf : Membership.mem (StructureGroupoid.maximalAtlas M' G') f
    xf : Membership.mem f.source (g x)
    h : And (ContinuousWithinAt g s x) (P (Function.comp (↑f) (Function.comp g ↑e. …
    h1 : ContinuousWithinAt (Function.comp g ↑e.symm) (Set.preimage (↑e.symm) s) ( …
    this : ContinuousAt (↑f) (Function.comp g (↑e.symm) (↑e x))
    ⊢ ContinuousWithinAt (Function.comp (↑f) (Function.comp g ↑e.symm)) (Set.preim …
  -/
  exact this.comp_continuousWithinAt h1
  /-
    🎉 no goals
  -/


theorem liftPropOn_indep_chart [HasGroupoid M G] [HasGroupoid M' G'] (he : e ∈ G.maximalAtlas M)
    (hf : f ∈ G'.maximalAtlas M') (h : LiftPropOn P g s) {y : H}
    (hy : y ∈ e.target ∩ e.symm ⁻¹' (s ∩ g ⁻¹' f.source)) :
    P (f ∘ g ∘ e.symm) (e.symm ⁻¹' s) y := by
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁷ : TopologicalSpace H
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    inst✝⁴ : TopologicalSpace H'
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    e : PartialHomeomorph M H
    f : PartialHomeomorph M' H'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s : Set M
    hG : G.LocalInvariantProp G' P
    inst✝¹ : HasGroupoid M G
    inst✝ : HasGroupoid M' G'
    he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
    hf : Membership.mem (StructureGroupoid.maximalAtlas M' G') f
    h : ChartedSpace.LiftPropOn P g s
    y : H
    hy : Membership.mem (Inter.inter e.target (Set.preimage (↑e.symm) (Inter.inter …
    ⊢ P (Function.comp (↑f) (Function.comp g ↑e.symm)) (Set.preimage (↑e.symm) s) y
  -/
  convert ((hG.liftPropWithinAt_indep_chart he (e.symm_mapsTo hy.1) hf hy.2.2).1 (h _ hy.2.1)).2
  /-
    case h.e'_3
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁷ : TopologicalSpace H
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    inst✝⁴ : TopologicalSpace H'
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    e : PartialHomeomorph M H
    f : PartialHomeomorph M' H'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s : Set M
    hG : G.LocalInvariantProp G' P
    inst✝¹ : HasGroupoid M G
    inst✝ : HasGroupoid M' G'
    he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
    hf : Membership.mem (StructureGroupoid.maximalAtlas M' G') f
    h : ChartedSpace.LiftPropOn P g s
    y : H
    hy : Membership.mem (Inter.inter e.target (Set.preimage (↑e.symm) (Inter.inter …
    ⊢ Eq y (↑e (↑e.symm y))
  -/
  rw [e.right_inv hy.1]
  /-
    🎉 no goals
  -/


theorem liftPropWithinAt_inter' (ht : t ∈ 𝓝[s] x) :
    LiftPropWithinAt P g (s ∩ t) x ↔ LiftPropWithinAt P g s x := by
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s t : Set M
    x : M
    hG : G.LocalInvariantProp G' P
    ht : Membership.mem (nhdsWithin x s) t
    ⊢ Iff (ChartedSpace.LiftPropWithinAt P g (Inter.inter s t) x) (ChartedSpace.Li …
  -/
  rw [liftPropWithinAt_iff', liftPropWithinAt_iff', continuousWithinAt_inter' ht, hG.congr_set]
  simp_rw [eventuallyEq_set, mem_preimage,
    (chartAt _ x).eventually_nhds' (fun x ↦ x ∈ s ∩ t ↔ x ∈ s) (mem_chart_source _ x)]
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s t : Set M
    x : M
    hG : G.LocalInvariantProp G' P
    ht : Membership.mem (nhdsWithin x s) t
    ⊢ Filter.Eventually (fun x => Iff (Membership.mem (Inter.inter s t) x) (Member …
  -/
  exact (mem_nhdsWithin_iff_eventuallyEq.mp ht).symm.mem_iff
  /-
    🎉 no goals
  -/


theorem liftPropWithinAt_inter (ht : t ∈ 𝓝 x) :
    LiftPropWithinAt P g (s ∩ t) x ↔ LiftPropWithinAt P g s x :=
  hG.liftPropWithinAt_inter' (mem_nhdsWithin_of_mem_nhds ht)


theorem liftPropWithinAt_congr_set (hu : s =ᶠ[𝓝 x] t) :
    LiftPropWithinAt P g s x ↔ LiftPropWithinAt P g t x := by
  rw [← hG.liftPropWithinAt_inter (s := s) hu, ← hG.liftPropWithinAt_inter (s := t) hu,
    ← eq_iff_iff]
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s t : Set M
    x : M
    hG : G.LocalInvariantProp G' P
    hu : (nhds x).EventuallyEq s t
    ⊢ Eq (ChartedSpace.LiftPropWithinAt P g (Inter.inter s (setOf fun x => (fun x  …
  -/
  congr 1
  /-
    case e_s
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s t : Set M
    x : M
    hG : G.LocalInvariantProp G' P
    hu : (nhds x).EventuallyEq s t
    ⊢ Eq (Inter.inter s (setOf fun x => (fun x => Eq (s x) (t x)) x)) (Inter.inter …
  -/
  aesop
  /-
    🎉 no goals
  -/


theorem liftPropAt_of_liftPropWithinAt (h : LiftPropWithinAt P g s x) (hs : s ∈ 𝓝 x) :
    LiftPropAt P g x := by
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s : Set M
    x : M
    hG : G.LocalInvariantProp G' P
    h : ChartedSpace.LiftPropWithinAt P g s x
    hs : Membership.mem (nhds x) s
    ⊢ ChartedSpace.LiftPropAt P g x
  -/
  rwa [← univ_inter s, hG.liftPropWithinAt_inter hs] at h
  /-
    🎉 no goals
  -/


theorem liftPropWithinAt_of_liftPropAt_of_mem_nhds (h : LiftPropAt P g x) (hs : s ∈ 𝓝 x) :
    LiftPropWithinAt P g s x := by
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s : Set M
    x : M
    hG : G.LocalInvariantProp G' P
    h : ChartedSpace.LiftPropAt P g x
    hs : Membership.mem (nhds x) s
    ⊢ ChartedSpace.LiftPropWithinAt P g s x
  -/
  rwa [← univ_inter s, hG.liftPropWithinAt_inter hs]
  /-
    🎉 no goals
  -/


theorem liftPropOn_of_locally_liftPropOn
    (h : ∀ x ∈ s, ∃ u, IsOpen u ∧ x ∈ u ∧ LiftPropOn P g (s ∩ u)) : LiftPropOn P g s := by
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s : Set M
    hG : G.LocalInvariantProp G' P
    h : ∀ (x : M), Membership.mem s x → Exists fun u => And (IsOpen u) (And (Membe …
    ⊢ ChartedSpace.LiftPropOn P g s
  -/
  intro x hx
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s : Set M
    hG : G.LocalInvariantProp G' P
    h : ∀ (x : M), Membership.mem s x → Exists fun u => And (IsOpen u) (And (Membe …
    x : M
    hx : Membership.mem s x
    ⊢ ChartedSpace.LiftPropWithinAt P g s x
  -/
  rcases h x hx with ⟨u, u_open, xu, hu⟩
  /-
    case intro.intro.intro
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s : Set M
    hG : G.LocalInvariantProp G' P
    h : ∀ (x : M), Membership.mem s x → Exists fun u => And (IsOpen u) (And (Membe …
    x : M
    hx : Membership.mem s x
    u : Set M
    u_open : IsOpen u
    xu : Membership.mem u x
    hu : ChartedSpace.LiftPropOn P g (Inter.inter s u)
    ⊢ ChartedSpace.LiftPropWithinAt P g s x
  -/
  have := hu x ⟨hx, xu⟩
  /-
    case intro.intro.intro
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s : Set M
    hG : G.LocalInvariantProp G' P
    h : ∀ (x : M), Membership.mem s x → Exists fun u => And (IsOpen u) (And (Membe …
    x : M
    hx : Membership.mem s x
    u : Set M
    u_open : IsOpen u
    xu : Membership.mem u x
    hu : ChartedSpace.LiftPropOn P g (Inter.inter s u)
    this : ChartedSpace.LiftPropWithinAt P g (Inter.inter s u) x
    ⊢ ChartedSpace.LiftPropWithinAt P g s x
  -/
  rwa [hG.liftPropWithinAt_inter] at this
  /-
    case intro.intro.intro
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s : Set M
    hG : G.LocalInvariantProp G' P
    h : ∀ (x : M), Membership.mem s x → Exists fun u => And (IsOpen u) (And (Membe …
    x : M
    hx : Membership.mem s x
    u : Set M
    u_open : IsOpen u
    xu : Membership.mem u x
    hu : ChartedSpace.LiftPropOn P g (Inter.inter s u)
    this : ChartedSpace.LiftPropWithinAt P g (Inter.inter s u) x
    ⊢ Membership.mem (nhds x) u
  -/
  exact u_open.mem_nhds xu
  /-
    🎉 no goals
  -/


theorem liftProp_of_locally_liftPropOn (h : ∀ x, ∃ u, IsOpen u ∧ x ∈ u ∧ LiftPropOn P g u) :
    LiftProp P g := by
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    hG : G.LocalInvariantProp G' P
    h : ∀ (x : M), Exists fun u => And (IsOpen u) (And (Membership.mem u x) (Chart …
    ⊢ ChartedSpace.LiftProp P g
  -/
  rw [← liftPropOn_univ]
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    hG : G.LocalInvariantProp G' P
    h : ∀ (x : M), Exists fun u => And (IsOpen u) (And (Membership.mem u x) (Chart …
    ⊢ ChartedSpace.LiftPropOn P g Set.univ
  -/
  refine hG.liftPropOn_of_locally_liftPropOn fun x _ ↦ ?_
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    hG : G.LocalInvariantProp G' P
    h : ∀ (x : M), Exists fun u => And (IsOpen u) (And (Membership.mem u x) (Chart …
    x : M
    x✝ : Membership.mem Set.univ x
    ⊢ Exists fun u => And (IsOpen u) (And (Membership.mem u x) (ChartedSpace.LiftP …
  -/
  simp [h x]
  /-
    🎉 no goals
  -/


theorem liftPropWithinAt_congr_of_eventuallyEq (h : LiftPropWithinAt P g s x) (h₁ : g' =ᶠ[𝓝[s] x] g)
    (hx : g' x = g x) : LiftPropWithinAt P g' s x := by
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    g g' : M → M'
    s : Set M
    x : M
    hG : G.LocalInvariantProp G' P
    h : ChartedSpace.LiftPropWithinAt P g s x
    h₁ : (nhdsWithin x s).EventuallyEq g' g
    hx : Eq (g' x) (g x)
    ⊢ ChartedSpace.LiftPropWithinAt P g' s x
  -/
  refine ⟨h.1.congr_of_eventuallyEq h₁ hx, ?_⟩
  refine hG.congr_nhdsWithin' ?_
    (by simp_rw [Function.comp_apply, (chartAt H x).left_inv (mem_chart_source H x), hx]) h.2
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    g g' : M → M'
    s : Set M
    x : M
    hG : G.LocalInvariantProp G' P
    h : ChartedSpace.LiftPropWithinAt P g s x
    h₁ : (nhdsWithin x s).EventuallyEq g' g
    hx : Eq (g' x) (g x)
    ⊢ (nhdsWithin (↑(chartAt H x) x) (Set.preimage (↑(chartAt H x).symm) s)).Event …
  -/
  simp_rw [EventuallyEq, Function.comp_apply]
  rw [(chartAt H x).eventually_nhdsWithin'
    (fun y ↦ chartAt H' (g' x) (g' y) = chartAt H' (g x) (g y)) (mem_chart_source H x)]
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    g g' : M → M'
    s : Set M
    x : M
    hG : G.LocalInvariantProp G' P
    h : ChartedSpace.LiftPropWithinAt P g s x
    h₁ : (nhdsWithin x s).EventuallyEq g' g
    hx : Eq (g' x) (g x)
    ⊢ Filter.Eventually (fun x_1 => Eq (↑(chartAt H' (g' x)) (g' x_1)) (↑(chartAt  …
  -/
  exact h₁.mono fun y hy ↦ by rw [hx, hy]
  /-
    🎉 no goals
  -/


theorem liftPropWithinAt_congr_of_eventuallyEq_of_mem (h : LiftPropWithinAt P g s x)
    (h₁ : g' =ᶠ[𝓝[s] x] g) (h₂ : x ∈ s) : LiftPropWithinAt P g' s x :=
  liftPropWithinAt_congr_of_eventuallyEq hG h h₁ (mem_of_mem_nhdsWithin h₂ h₁ : _)


theorem liftPropWithinAt_congr_iff_of_eventuallyEq (h₁ : g' =ᶠ[𝓝[s] x] g) (hx : g' x = g x) :
    LiftPropWithinAt P g' s x ↔ LiftPropWithinAt P g s x :=
  ⟨fun h ↦ hG.liftPropWithinAt_congr_of_eventuallyEq h h₁.symm hx.symm,
    fun h ↦ hG.liftPropWithinAt_congr_of_eventuallyEq h h₁ hx⟩


theorem liftPropWithinAt_congr_iff (h₁ : ∀ y ∈ s, g' y = g y) (hx : g' x = g x) :
    LiftPropWithinAt P g' s x ↔ LiftPropWithinAt P g s x :=
  hG.liftPropWithinAt_congr_iff_of_eventuallyEq (eventually_nhdsWithin_of_forall h₁) hx


theorem liftPropWithinAt_congr_iff_of_mem (h₁ : ∀ y ∈ s, g' y = g y) (hx : x ∈ s) :
    LiftPropWithinAt P g' s x ↔ LiftPropWithinAt P g s x :=
  hG.liftPropWithinAt_congr_iff_of_eventuallyEq (eventually_nhdsWithin_of_forall h₁) (h₁ _ hx)


theorem liftPropWithinAt_congr (h : LiftPropWithinAt P g s x) (h₁ : ∀ y ∈ s, g' y = g y)
    (hx : g' x = g x) : LiftPropWithinAt P g' s x :=
  (hG.liftPropWithinAt_congr_iff h₁ hx).mpr h


theorem liftPropWithinAt_congr_of_mem (h : LiftPropWithinAt P g s x) (h₁ : ∀ y ∈ s, g' y = g y)
    (hx : x ∈ s) : LiftPropWithinAt P g' s x :=
  (hG.liftPropWithinAt_congr_iff h₁ (h₁ _ hx)).mpr h


theorem liftPropAt_congr_iff_of_eventuallyEq (h₁ : g' =ᶠ[𝓝 x] g) :
    LiftPropAt P g' x ↔ LiftPropAt P g x :=
                                                    /-
                                                      H : Type u_1
                                                      M : Type u_2
                                                      H' : Type u_3
                                                      M' : Type u_4
                                                      inst✝⁵ : TopologicalSpace H
                                                      inst✝⁴ : TopologicalSpace M
                                                      inst✝³ : ChartedSpace H M
                                                      inst✝² : TopologicalSpace H'
                                                      inst✝¹ : TopologicalSpace M'
                                                      inst✝ : ChartedSpace H' M'
                                                      G : StructureGroupoid H
                                                      G' : StructureGroupoid H'
                                                      P : (H → H') → Set H → H → Prop
                                                      g g' : M → M'
                                                      x : M
                                                      hG : G.LocalInvariantProp G' P
                                                      h₁ : (nhds x).EventuallyEq g' g
                                                      ⊢ (nhdsWithin x Set.univ).EventuallyEq g' g
                                                    -/
  hG.liftPropWithinAt_congr_iff_of_eventuallyEq (by simp_rw [nhdsWithin_univ, h₁]) h₁.eq_of_nhds
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem liftPropAt_congr_of_eventuallyEq (h : LiftPropAt P g x) (h₁ : g' =ᶠ[𝓝 x] g) :
    LiftPropAt P g' x :=
  (hG.liftPropAt_congr_iff_of_eventuallyEq h₁).mpr h


theorem liftPropOn_congr (h : LiftPropOn P g s) (h₁ : ∀ y ∈ s, g' y = g y) : LiftPropOn P g' s :=
  fun x hx ↦ hG.liftPropWithinAt_congr (h x hx) h₁ (h₁ x hx)


theorem liftPropOn_congr_iff (h₁ : ∀ y ∈ s, g' y = g y) : LiftPropOn P g' s ↔ LiftPropOn P g s :=
  ⟨fun h ↦ hG.liftPropOn_congr h fun y hy ↦ (h₁ y hy).symm, fun h ↦ hG.liftPropOn_congr h h₁⟩


theorem liftPropWithinAt_mono_of_mem_nhdsWithin
    (mono_of_mem_nhdsWithin : ∀ ⦃s x t⦄ ⦃f : H → H'⦄, s ∈ 𝓝[t] x → P f s x → P f t x)
    (h : LiftPropWithinAt P g s x) (hst : s ∈ 𝓝[t] x) : LiftPropWithinAt P g t x := by
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s t : Set M
    x : M
    mono_of_mem_nhdsWithin : ∀ ⦃s : Set H⦄ ⦃x : H⦄ ⦃t : Set H⦄ ⦃f : H → H'⦄, Membe …
    h : ChartedSpace.LiftPropWithinAt P g s x
    hst : Membership.mem (nhdsWithin x t) s
    ⊢ ChartedSpace.LiftPropWithinAt P g t x
  -/
  simp only [liftPropWithinAt_iff'] at h ⊢
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s t : Set M
    x : M
    mono_of_mem_nhdsWithin : ∀ ⦃s : Set H⦄ ⦃x : H⦄ ⦃t : Set H⦄ ⦃f : H → H'⦄, Membe …
    hst : Membership.mem (nhdsWithin x t) s
    h : And (ContinuousWithinAt g s x) (P (Function.comp (↑(chartAt H' (g x))) (Fu …
    ⊢ And (ContinuousWithinAt g t x) (P (Function.comp (↑(chartAt H' (g x))) (Func …
  -/
  refine ⟨h.1.mono_of_mem_nhdsWithin hst, mono_of_mem_nhdsWithin ?_ h.2⟩
  simp_rw [← mem_map, (chartAt H x).symm.map_nhdsWithin_preimage_eq (mem_chart_target H x),
    (chartAt H x).left_inv (mem_chart_source H x), hst]


@[deprecated (since := "2024-10-31")]
alias liftPropWithinAt_mono_of_mem := liftPropWithinAt_mono_of_mem_nhdsWithin


theorem liftPropWithinAt_mono (mono : ∀ ⦃s x t⦄ ⦃f : H → H'⦄, t ⊆ s → P f s x → P f t x)
    (h : LiftPropWithinAt P g s x) (hts : t ⊆ s) : LiftPropWithinAt P g t x := by
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s t : Set M
    x : M
    mono : ∀ ⦃s : Set H⦄ ⦃x : H⦄ ⦃t : Set H⦄ ⦃f : H → H'⦄, HasSubset.Subset t s →  …
    h : ChartedSpace.LiftPropWithinAt P g s x
    hts : HasSubset.Subset t s
    ⊢ ChartedSpace.LiftPropWithinAt P g t x
  -/
  refine ⟨h.1.mono hts, mono (fun y hy ↦ ?_) h.2⟩
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s t : Set M
    x : M
    mono : ∀ ⦃s : Set H⦄ ⦃x : H⦄ ⦃t : Set H⦄ ⦃f : H → H'⦄, HasSubset.Subset t s →  …
    h : ChartedSpace.LiftPropWithinAt P g s x
    hts : HasSubset.Subset t s
    y : H
    hy : Membership.mem (Set.preimage (↑(chartAt H x).symm) t) y
    ⊢ Membership.mem (Set.preimage (↑(chartAt H x).symm) s) y
  -/
  simp only [mfld_simps] at hy
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s t : Set M
    x : M
    mono : ∀ ⦃s : Set H⦄ ⦃x : H⦄ ⦃t : Set H⦄ ⦃f : H → H'⦄, HasSubset.Subset t s →  …
    h : ChartedSpace.LiftPropWithinAt P g s x
    hts : HasSubset.Subset t s
    y : H
    hy : Membership.mem t (↑(chartAt H x).symm y)
    ⊢ Membership.mem (Set.preimage (↑(chartAt H x).symm) s) y
  -/
  simp only [hy, hts _, mfld_simps]
  /-
    🎉 no goals
  -/


theorem liftPropWithinAt_of_liftPropAt (mono : ∀ ⦃s x t⦄ ⦃f : H → H'⦄, t ⊆ s → P f s x → P f t x)
    (h : LiftPropAt P g x) : LiftPropWithinAt P g s x := by
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s : Set M
    x : M
    mono : ∀ ⦃s : Set H⦄ ⦃x : H⦄ ⦃t : Set H⦄ ⦃f : H → H'⦄, HasSubset.Subset t s →  …
    h : ChartedSpace.LiftPropAt P g x
    ⊢ ChartedSpace.LiftPropWithinAt P g s x
  -/
  rw [← liftPropWithinAt_univ] at h
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s : Set M
    x : M
    mono : ∀ ⦃s : Set H⦄ ⦃x : H⦄ ⦃t : Set H⦄ ⦃f : H → H'⦄, HasSubset.Subset t s →  …
    h : ChartedSpace.LiftPropWithinAt P g Set.univ x
    ⊢ ChartedSpace.LiftPropWithinAt P g s x
  -/
  exact liftPropWithinAt_mono mono h (subset_univ _)
  /-
    🎉 no goals
  -/


theorem liftPropOn_mono (mono : ∀ ⦃s x t⦄ ⦃f : H → H'⦄, t ⊆ s → P f s x → P f t x)
    (h : LiftPropOn P g t) (hst : s ⊆ t) : LiftPropOn P g s :=
  fun x hx ↦ liftPropWithinAt_mono mono (h x (hst hx)) hst


theorem liftPropOn_of_liftProp (mono : ∀ ⦃s x t⦄ ⦃f : H → H'⦄, t ⊆ s → P f s x → P f t x)
    (h : LiftProp P g) : LiftPropOn P g s := by
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s : Set M
    mono : ∀ ⦃s : Set H⦄ ⦃x : H⦄ ⦃t : Set H⦄ ⦃f : H → H'⦄, HasSubset.Subset t s →  …
    h : ChartedSpace.LiftProp P g
    ⊢ ChartedSpace.LiftPropOn P g s
  -/
  rw [← liftPropOn_univ] at h
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    P : (H → H') → Set H → H → Prop
    g : M → M'
    s : Set M
    mono : ∀ ⦃s : Set H⦄ ⦃x : H⦄ ⦃t : Set H⦄ ⦃f : H → H'⦄, HasSubset.Subset t s →  …
    h : ChartedSpace.LiftPropOn P g Set.univ
    ⊢ ChartedSpace.LiftPropOn P g s
  -/
  exact liftPropOn_mono mono h (subset_univ _)
  /-
    🎉 no goals
  -/


theorem liftPropAt_of_mem_maximalAtlas [HasGroupoid M G] (hG : G.LocalInvariantProp G Q)
    (hQ : ∀ y, Q id univ y) (he : e ∈ maximalAtlas M G) (hx : x ∈ e.source) : LiftPropAt Q e x := by
  simp_rw [LiftPropAt, hG.liftPropWithinAt_indep_chart he hx G.id_mem_maximalAtlas (mem_univ _),
    (e.continuousAt hx).continuousWithinAt, true_and]
  /-
    H : Type u_1
    M : Type u_2
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    G : StructureGroupoid H
    e : PartialHomeomorph M H
    x : M
    Q : (H → H) → Set H → H → Prop
    inst✝ : HasGroupoid M G
    hG : G.LocalInvariantProp G Q
    hQ : ∀ (y : H), Q id Set.univ y
    he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
    hx : Membership.mem e.source x
    ⊢ Q (Function.comp (↑(PartialHomeomorph.refl H)) (Function.comp ↑e ↑e.symm)) ( …
  -/
  exact hG.congr' (e.eventually_right_inverse' hx) (hQ _)
  /-
    🎉 no goals
  -/


theorem liftPropOn_of_mem_maximalAtlas [HasGroupoid M G] (hG : G.LocalInvariantProp G Q)
    (hQ : ∀ y, Q id univ y) (he : e ∈ maximalAtlas M G) : LiftPropOn Q e e.source := by
  /-
    H : Type u_1
    M : Type u_2
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    G : StructureGroupoid H
    e : PartialHomeomorph M H
    Q : (H → H) → Set H → H → Prop
    inst✝ : HasGroupoid M G
    hG : G.LocalInvariantProp G Q
    hQ : ∀ (y : H), Q id Set.univ y
    he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
    ⊢ ChartedSpace.LiftPropOn Q (↑e) e.source
  -/
  intro x hx
  /-
    H : Type u_1
    M : Type u_2
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    G : StructureGroupoid H
    e : PartialHomeomorph M H
    Q : (H → H) → Set H → H → Prop
    inst✝ : HasGroupoid M G
    hG : G.LocalInvariantProp G Q
    hQ : ∀ (y : H), Q id Set.univ y
    he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
    x : M
    hx : Membership.mem e.source x
    ⊢ ChartedSpace.LiftPropWithinAt Q (↑e) e.source x
  -/
  apply hG.liftPropWithinAt_of_liftPropAt_of_mem_nhds (hG.liftPropAt_of_mem_maximalAtlas hQ he hx)
  /-
    H : Type u_1
    M : Type u_2
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    G : StructureGroupoid H
    e : PartialHomeomorph M H
    Q : (H → H) → Set H → H → Prop
    inst✝ : HasGroupoid M G
    hG : G.LocalInvariantProp G Q
    hQ : ∀ (y : H), Q id Set.univ y
    he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
    x : M
    hx : Membership.mem e.source x
    ⊢ Membership.mem (nhds x) e.source
  -/
  exact e.open_source.mem_nhds hx
  /-
    🎉 no goals
  -/


theorem liftPropAt_symm_of_mem_maximalAtlas [HasGroupoid M G] {x : H}
    (hG : G.LocalInvariantProp G Q) (hQ : ∀ y, Q id univ y) (he : e ∈ maximalAtlas M G)
    (hx : x ∈ e.target) : LiftPropAt Q e.symm x := by
  suffices h : Q (e ∘ e.symm) univ x by
    have : e.symm x ∈ e.source := by simp only [hx, mfld_simps]
    rw [LiftPropAt, hG.liftPropWithinAt_indep_chart G.id_mem_maximalAtlas (mem_univ _) he this]
    refine ⟨(e.symm.continuousAt hx).continuousWithinAt, ?_⟩
    simp only [h, mfld_simps]
  /-
    H : Type u_1
    M : Type u_2
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    G : StructureGroupoid H
    e : PartialHomeomorph M H
    Q : (H → H) → Set H → H → Prop
    inst✝ : HasGroupoid M G
    x : H
    hG : G.LocalInvariantProp G Q
    hQ : ∀ (y : H), Q id Set.univ y
    he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
    hx : Membership.mem e.target x
    ⊢ Q (Function.comp ↑e ↑e.symm) Set.univ x
  -/
  exact hG.congr' (e.eventually_right_inverse hx) (hQ x)
  /-
    🎉 no goals
  -/


theorem liftPropOn_symm_of_mem_maximalAtlas [HasGroupoid M G] (hG : G.LocalInvariantProp G Q)
    (hQ : ∀ y, Q id univ y) (he : e ∈ maximalAtlas M G) : LiftPropOn Q e.symm e.target := by
  /-
    H : Type u_1
    M : Type u_2
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    G : StructureGroupoid H
    e : PartialHomeomorph M H
    Q : (H → H) → Set H → H → Prop
    inst✝ : HasGroupoid M G
    hG : G.LocalInvariantProp G Q
    hQ : ∀ (y : H), Q id Set.univ y
    he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
    ⊢ ChartedSpace.LiftPropOn Q (↑e.symm) e.target
  -/
  intro x hx
  apply hG.liftPropWithinAt_of_liftPropAt_of_mem_nhds
    (hG.liftPropAt_symm_of_mem_maximalAtlas hQ he hx)
  /-
    H : Type u_1
    M : Type u_2
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    G : StructureGroupoid H
    e : PartialHomeomorph M H
    Q : (H → H) → Set H → H → Prop
    inst✝ : HasGroupoid M G
    hG : G.LocalInvariantProp G Q
    hQ : ∀ (y : H), Q id Set.univ y
    he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
    x : H
    hx : Membership.mem e.target x
    ⊢ Membership.mem (nhds x) e.target
  -/
  exact e.open_target.mem_nhds hx
  /-
    🎉 no goals
  -/


theorem liftPropAt_chart [HasGroupoid M G] (hG : G.LocalInvariantProp G Q) (hQ : ∀ y, Q id univ y) :
    LiftPropAt Q (chartAt (H := H) x) x :=
  hG.liftPropAt_of_mem_maximalAtlas hQ (chart_mem_maximalAtlas G x) (mem_chart_source H x)


theorem liftPropOn_chart [HasGroupoid M G] (hG : G.LocalInvariantProp G Q) (hQ : ∀ y, Q id univ y) :
    LiftPropOn Q (chartAt (H := H) x) (chartAt (H := H) x).source :=
  hG.liftPropOn_of_mem_maximalAtlas hQ (chart_mem_maximalAtlas G x)


theorem liftPropAt_chart_symm [HasGroupoid M G] (hG : G.LocalInvariantProp G Q)
    (hQ : ∀ y, Q id univ y) : LiftPropAt Q (chartAt (H := H) x).symm ((chartAt H x) x) :=
                                                                             /-
                                                                               H : Type u_1
                                                                               M : Type u_2
                                                                               inst✝³ : TopologicalSpace H
                                                                               inst✝² : TopologicalSpace M
                                                                               inst✝¹ : ChartedSpace H M
                                                                               G : StructureGroupoid H
                                                                               x : M
                                                                               Q : (H → H) → Set H → H → Prop
                                                                               inst✝ : HasGroupoid M G
                                                                               hG : G.LocalInvariantProp G Q
                                                                               hQ : ∀ (y : H), Q id Set.univ y
                                                                               ⊢ Membership.mem (chartAt H x).target (↑(chartAt H x) x)
                                                                             -/
  hG.liftPropAt_symm_of_mem_maximalAtlas hQ (chart_mem_maximalAtlas G x) (by simp)
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem liftPropOn_chart_symm [HasGroupoid M G] (hG : G.LocalInvariantProp G Q)
    (hQ : ∀ y, Q id univ y) : LiftPropOn Q (chartAt (H := H) x).symm (chartAt H x).target :=
  hG.liftPropOn_symm_of_mem_maximalAtlas hQ (chart_mem_maximalAtlas G x)


theorem liftPropAt_of_mem_groupoid (hG : G.LocalInvariantProp G Q) (hQ : ∀ y, Q id univ y)
    {f : PartialHomeomorph H H} (hf : f ∈ G) {x : H} (hx : x ∈ f.source) : LiftPropAt Q f x :=
  liftPropAt_of_mem_maximalAtlas hG hQ (G.mem_maximalAtlas_of_mem_groupoid hf) hx


theorem liftPropOn_of_mem_groupoid (hG : G.LocalInvariantProp G Q) (hQ : ∀ y, Q id univ y)
    {f : PartialHomeomorph H H} (hf : f ∈ G) : LiftPropOn Q f f.source :=
  liftPropOn_of_mem_maximalAtlas hG hQ (G.mem_maximalAtlas_of_mem_groupoid hf)


theorem liftProp_id (hG : G.LocalInvariantProp G Q) (hQ : ∀ y, Q id univ y) :
    LiftProp Q (id : M → M) := by
  /-
    H : Type u_1
    M : Type u_2
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    G : StructureGroupoid H
    Q : (H → H) → Set H → H → Prop
    hG : G.LocalInvariantProp G Q
    hQ : ∀ (y : H), Q id Set.univ y
    ⊢ ChartedSpace.LiftProp Q id
  -/
  simp_rw [liftProp_iff, continuous_id, true_and]
  /-
    H : Type u_1
    M : Type u_2
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    G : StructureGroupoid H
    Q : (H → H) → Set H → H → Prop
    hG : G.LocalInvariantProp G Q
    hQ : ∀ (y : H), Q id Set.univ y
    ⊢ ∀ (x : M), Q (Function.comp (↑(chartAt H (id x))) (Function.comp id ↑(chartA …
  -/
  exact fun x ↦ hG.congr' ((chartAt H x).eventually_right_inverse <| mem_chart_target H x) (hQ _)
  /-
    🎉 no goals
  -/


theorem liftPropAt_iff_comp_subtype_val (hG : LocalInvariantProp G G' P) {U : Opens M}
    (f : M → M') (x : U) :
    LiftPropAt P f x ↔ LiftPropAt P (f ∘ Subtype.val) x := by
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    hG : G.LocalInvariantProp G' P
    U : TopologicalSpace.Opens M
    f : M → M'
    x : Subtype fun x => Membership.mem U x
    ⊢ Iff (ChartedSpace.LiftPropAt P f ↑x) (ChartedSpace.LiftPropAt P (Function.co …
  -/
  simp only [LiftPropAt, liftPropWithinAt_iff']
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    hG : G.LocalInvariantProp G' P
    U : TopologicalSpace.Opens M
    f : M → M'
    x : Subtype fun x => Membership.mem U x
    ⊢ Iff (And (ContinuousWithinAt f Set.univ ↑x) (P (Function.comp (↑(chartAt H'  …
  -/
  congrm ?_ ∧ ?_
    /-
      case a.refine_1
      H : Type u_1
      M : Type u_2
      H' : Type u_3
      M' : Type u_4
      inst✝⁵ : TopologicalSpace H
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : TopologicalSpace H'
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H' M'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      P : (H → H') → Set H → H → Prop
      hG : G.LocalInvariantProp G' P
      U : TopologicalSpace.Opens M
      f : M → M'
      x : Subtype fun x => Membership.mem U x
      ⊢ Iff (ContinuousWithinAt f Set.univ ↑x) (ContinuousWithinAt (Function.comp f  …
    -/
  · simp_rw [continuousWithinAt_univ, U.isOpenEmbedding'.continuousAt_iff]
    /-
      🎉 no goals
    -/
    /-
      case a.refine_2
      H : Type u_1
      M : Type u_2
      H' : Type u_3
      M' : Type u_4
      inst✝⁵ : TopologicalSpace H
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : TopologicalSpace H'
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H' M'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      P : (H → H') → Set H → H → Prop
      hG : G.LocalInvariantProp G' P
      U : TopologicalSpace.Opens M
      f : M → M'
      x : Subtype fun x => Membership.mem U x
      ⊢ Iff (P (Function.comp (↑(chartAt H' (f ↑x))) (Function.comp f ↑(chartAt H ↑x …
    -/
  · apply hG.congr_iff
    /-
      case a.refine_2
      H : Type u_1
      M : Type u_2
      H' : Type u_3
      M' : Type u_4
      inst✝⁵ : TopologicalSpace H
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : TopologicalSpace H'
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H' M'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      P : (H → H') → Set H → H → Prop
      hG : G.LocalInvariantProp G' P
      U : TopologicalSpace.Opens M
      f : M → M'
      x : Subtype fun x => Membership.mem U x
      ⊢ (nhds (↑(chartAt H ↑x) ↑x)).EventuallyEq (Function.comp (↑(chartAt H' (f ↑x) …
    -/
    exact (U.chartAt_subtype_val_symm_eventuallyEq).fun_comp (chartAt H' (f x) ∘ f)
    /-
      🎉 no goals
    -/


theorem liftPropAt_iff_comp_inclusion (hG : LocalInvariantProp G G' P) {U V : Opens M} (hUV : U ≤ V)
    (f : V → M') (x : U) :
    LiftPropAt P f (Set.inclusion hUV x) ↔ LiftPropAt P (f ∘ Set.inclusion hUV : U → M') x := by
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    hG : G.LocalInvariantProp G' P
    U V : TopologicalSpace.Opens M
    hUV : LE.le U V
    f : (Subtype fun x => Membership.mem V x) → M'
    x : Subtype fun x => Membership.mem U x
    ⊢ Iff (ChartedSpace.LiftPropAt P f (Set.inclusion hUV x)) (ChartedSpace.LiftPr …
  -/
  simp only [LiftPropAt, liftPropWithinAt_iff']
  /-
    H : Type u_1
    M : Type u_2
    H' : Type u_3
    M' : Type u_4
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : TopologicalSpace H'
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    G : StructureGroupoid H
    G' : StructureGroupoid H'
    P : (H → H') → Set H → H → Prop
    hG : G.LocalInvariantProp G' P
    U V : TopologicalSpace.Opens M
    hUV : LE.le U V
    f : (Subtype fun x => Membership.mem V x) → M'
    x : Subtype fun x => Membership.mem U x
    ⊢ Iff (And (ContinuousWithinAt f Set.univ (Set.inclusion hUV x)) (P (Function. …
  -/
  congrm ?_ ∧ ?_
  · simp_rw [continuousWithinAt_univ,
      (TopologicalSpace.Opens.isOpenEmbedding_of_le hUV).continuousAt_iff]
    /-
      case a.refine_2
      H : Type u_1
      M : Type u_2
      H' : Type u_3
      M' : Type u_4
      inst✝⁵ : TopologicalSpace H
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : TopologicalSpace H'
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H' M'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      P : (H → H') → Set H → H → Prop
      hG : G.LocalInvariantProp G' P
      U V : TopologicalSpace.Opens M
      hUV : LE.le U V
      f : (Subtype fun x => Membership.mem V x) → M'
      x : Subtype fun x => Membership.mem U x
      ⊢ Iff (P (Function.comp (↑(chartAt H' (f (Set.inclusion hUV x)))) (Function.co …
    -/
  · apply hG.congr_iff
    exact (TopologicalSpace.Opens.chartAt_inclusion_symm_eventuallyEq hUV).fun_comp
      (chartAt H' (f (Set.inclusion hUV x)) ∘ f)


theorem liftProp_subtype_val {Q : (H → H) → Set H → H → Prop} (hG : LocalInvariantProp G G Q)
    (hQ : ∀ y, Q id univ y) (U : Opens M) :
    LiftProp Q (Subtype.val : U → M) := by
  /-
    H : Type u_1
    M : Type u_2
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    G : StructureGroupoid H
    Q : (H → H) → Set H → H → Prop
    hG : G.LocalInvariantProp G Q
    hQ : ∀ (y : H), Q id Set.univ y
    U : TopologicalSpace.Opens M
    ⊢ ChartedSpace.LiftProp Q Subtype.val
  -/
  intro x
  /-
    H : Type u_1
    M : Type u_2
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    G : StructureGroupoid H
    Q : (H → H) → Set H → H → Prop
    hG : G.LocalInvariantProp G Q
    hQ : ∀ (y : H), Q id Set.univ y
    U : TopologicalSpace.Opens M
    x : Subtype fun x => Membership.mem U x
    ⊢ ChartedSpace.LiftPropAt Q Subtype.val x
  -/
  show LiftPropAt Q (id ∘ Subtype.val) x
  /-
    H : Type u_1
    M : Type u_2
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    G : StructureGroupoid H
    Q : (H → H) → Set H → H → Prop
    hG : G.LocalInvariantProp G Q
    hQ : ∀ (y : H), Q id Set.univ y
    U : TopologicalSpace.Opens M
    x : Subtype fun x => Membership.mem U x
    ⊢ ChartedSpace.LiftPropAt Q (Function.comp id Subtype.val) x
  -/
  rw [← hG.liftPropAt_iff_comp_subtype_val]
  /-
    H : Type u_1
    M : Type u_2
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    G : StructureGroupoid H
    Q : (H → H) → Set H → H → Prop
    hG : G.LocalInvariantProp G Q
    hQ : ∀ (y : H), Q id Set.univ y
    U : TopologicalSpace.Opens M
    x : Subtype fun x => Membership.mem U x
    ⊢ ChartedSpace.LiftPropAt Q id ↑x
  -/
  apply hG.liftProp_id hQ
  /-
    🎉 no goals
  -/


theorem liftProp_inclusion {Q : (H → H) → Set H → H → Prop} (hG : LocalInvariantProp G G Q)
    (hQ : ∀ y, Q id univ y) {U V : Opens M} (hUV : U ≤ V) :
    LiftProp Q (Opens.inclusion hUV : U → V) := by
  /-
    H : Type u_1
    M : Type u_2
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    G : StructureGroupoid H
    Q : (H → H) → Set H → H → Prop
    hG : G.LocalInvariantProp G Q
    hQ : ∀ (y : H), Q id Set.univ y
    U V : TopologicalSpace.Opens M
    hUV : LE.le U V
    ⊢ ChartedSpace.LiftProp Q (TopologicalSpace.Opens.inclusion hUV)
  -/
  intro x
  /-
    H : Type u_1
    M : Type u_2
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    G : StructureGroupoid H
    Q : (H → H) → Set H → H → Prop
    hG : G.LocalInvariantProp G Q
    hQ : ∀ (y : H), Q id Set.univ y
    U V : TopologicalSpace.Opens M
    hUV : LE.le U V
    x : Subtype fun x => Membership.mem U x
    ⊢ ChartedSpace.LiftPropAt Q (TopologicalSpace.Opens.inclusion hUV) x
  -/
  show LiftPropAt Q (id ∘ Opens.inclusion hUV) x
  /-
    H : Type u_1
    M : Type u_2
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    G : StructureGroupoid H
    Q : (H → H) → Set H → H → Prop
    hG : G.LocalInvariantProp G Q
    hQ : ∀ (y : H), Q id Set.univ y
    U V : TopologicalSpace.Opens M
    hUV : LE.le U V
    x : Subtype fun x => Membership.mem U x
    ⊢ ChartedSpace.LiftPropAt Q (Function.comp id (TopologicalSpace.Opens.inclusio …
  -/
  rw [← hG.liftPropAt_iff_comp_inclusion hUV]
  /-
    H : Type u_1
    M : Type u_2
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    G : StructureGroupoid H
    Q : (H → H) → Set H → H → Prop
    hG : G.LocalInvariantProp G Q
    hQ : ∀ (y : H), Q id Set.univ y
    U V : TopologicalSpace.Opens M
    hUV : LE.le U V
    x : Subtype fun x => Membership.mem U x
    ⊢ ChartedSpace.LiftPropAt Q id (Set.inclusion hUV x)
  -/
  apply hG.liftProp_id hQ
  /-
    🎉 no goals
  -/


/-- A function from a model space `H` to itself is a local structomorphism, with respect to a
structure groupoid `G` for `H`, relative to a set `s` in `H`, if for all points `x` in the set, the
function agrees with a `G`-structomorphism on `s` in a neighbourhood of `x`. -/
def IsLocalStructomorphWithinAt (f : H → H) (s : Set H) (x : H) : Prop :=
  x ∈ s → ∃ e : PartialHomeomorph H H, e ∈ G ∧ EqOn f e.toFun (s ∩ e.source) ∧ x ∈ e.source


/-- For a groupoid `G` which is `ClosedUnderRestriction`, being a local structomorphism is a local
invariant property. -/
theorem isLocalStructomorphWithinAt_localInvariantProp [ClosedUnderRestriction G] :
    LocalInvariantProp G G (IsLocalStructomorphWithinAt G) :=
  { is_local := by
      /-
        H : Type u_1
        inst✝¹ : TopologicalSpace H
        G : StructureGroupoid H
        inst✝ : ClosedUnderRestriction G
        ⊢ ∀ {s : Set H} {x : H} {u : Set H} {f : H → H}, IsOpen u → Membership.mem u x …
      -/
      intro s x u f hu hux
      /-
        H : Type u_1
        inst✝¹ : TopologicalSpace H
        G : StructureGroupoid H
        inst✝ : ClosedUnderRestriction G
        s : Set H
        x : H
        u : Set H
        f : H → H
        hu : IsOpen u
        hux : Membership.mem u x
        ⊢ Iff (G.IsLocalStructomorphWithinAt f s x) (G.IsLocalStructomorphWithinAt f ( …
      -/
      constructor
        /-
          case mp
          H : Type u_1
          inst✝¹ : TopologicalSpace H
          G : StructureGroupoid H
          inst✝ : ClosedUnderRestriction G
          s : Set H
          x : H
          u : Set H
          f : H → H
          hu : IsOpen u
          hux : Membership.mem u x
          ⊢ G.IsLocalStructomorphWithinAt f s x → G.IsLocalStructomorphWithinAt f (Inter …
        -/
      · rintro h hx
        /-
          case mp
          H : Type u_1
          inst✝¹ : TopologicalSpace H
          G : StructureGroupoid H
          inst✝ : ClosedUnderRestriction G
          s : Set H
          x : H
          u : Set H
          f : H → H
          hu : IsOpen u
          hux : Membership.mem u x
          h : G.IsLocalStructomorphWithinAt f s x
          hx : Membership.mem (Inter.inter s u) x
          ⊢ Exists fun e => And (Membership.mem G e) (And (Set.EqOn f (↑e.toPartialEquiv …
        -/
        rcases h hx.1 with ⟨e, heG, hef, hex⟩
        /-
          case mp.intro.intro.intro
          H : Type u_1
          inst✝¹ : TopologicalSpace H
          G : StructureGroupoid H
          inst✝ : ClosedUnderRestriction G
          s : Set H
          x : H
          u : Set H
          f : H → H
          hu : IsOpen u
          hux : Membership.mem u x
          h : G.IsLocalStructomorphWithinAt f s x
          hx : Membership.mem (Inter.inter s u) x
          e : PartialHomeomorph H H
          heG : Membership.mem G e
          hef : Set.EqOn f (↑e.toPartialEquiv) (Inter.inter s e.source)
          hex : Membership.mem e.source x
          ⊢ Exists fun e => And (Membership.mem G e) (And (Set.EqOn f (↑e.toPartialEquiv …
        -/
        have : s ∩ u ∩ e.source ⊆ s ∩ e.source := by mfld_set_tac
        /-
          case mp.intro.intro.intro
          H : Type u_1
          inst✝¹ : TopologicalSpace H
          G : StructureGroupoid H
          inst✝ : ClosedUnderRestriction G
          s : Set H
          x : H
          u : Set H
          f : H → H
          hu : IsOpen u
          hux : Membership.mem u x
          h : G.IsLocalStructomorphWithinAt f s x
          hx : Membership.mem (Inter.inter s u) x
          e : PartialHomeomorph H H
          heG : Membership.mem G e
          hef : Set.EqOn f (↑e.toPartialEquiv) (Inter.inter s e.source)
          hex : Membership.mem e.source x
          this : HasSubset.Subset (Inter.inter (Inter.inter s u) e.source) (Inter.inter  …
          ⊢ Exists fun e => And (Membership.mem G e) (And (Set.EqOn f (↑e.toPartialEquiv …
        -/
        exact ⟨e, heG, hef.mono this, hex⟩
        /-
          🎉 no goals
        -/
        /-
          case mpr
          H : Type u_1
          inst✝¹ : TopologicalSpace H
          G : StructureGroupoid H
          inst✝ : ClosedUnderRestriction G
          s : Set H
          x : H
          u : Set H
          f : H → H
          hu : IsOpen u
          hux : Membership.mem u x
          ⊢ G.IsLocalStructomorphWithinAt f (Inter.inter s u) x → G.IsLocalStructomorphW …
        -/
      · rintro h hx
        /-
          case mpr
          H : Type u_1
          inst✝¹ : TopologicalSpace H
          G : StructureGroupoid H
          inst✝ : ClosedUnderRestriction G
          s : Set H
          x : H
          u : Set H
          f : H → H
          hu : IsOpen u
          hux : Membership.mem u x
          h : G.IsLocalStructomorphWithinAt f (Inter.inter s u) x
          hx : Membership.mem s x
          ⊢ Exists fun e => And (Membership.mem G e) (And (Set.EqOn f (↑e.toPartialEquiv …
        -/
        rcases h ⟨hx, hux⟩ with ⟨e, heG, hef, hex⟩
        /-
          case mpr.intro.intro.intro
          H : Type u_1
          inst✝¹ : TopologicalSpace H
          G : StructureGroupoid H
          inst✝ : ClosedUnderRestriction G
          s : Set H
          x : H
          u : Set H
          f : H → H
          hu : IsOpen u
          hux : Membership.mem u x
          h : G.IsLocalStructomorphWithinAt f (Inter.inter s u) x
          hx : Membership.mem s x
          e : PartialHomeomorph H H
          heG : Membership.mem G e
          hef : Set.EqOn f (↑e.toPartialEquiv) (Inter.inter (Inter.inter s u) e.source)
          hex : Membership.mem e.source x
          ⊢ Exists fun e => And (Membership.mem G e) (And (Set.EqOn f (↑e.toPartialEquiv …
        -/
        refine ⟨e.restr (interior u), ?_, ?_, ?_⟩
          /-
            case mpr.intro.intro.intro.refine_1
            H : Type u_1
            inst✝¹ : TopologicalSpace H
            G : StructureGroupoid H
            inst✝ : ClosedUnderRestriction G
            s : Set H
            x : H
            u : Set H
            f : H → H
            hu : IsOpen u
            hux : Membership.mem u x
            h : G.IsLocalStructomorphWithinAt f (Inter.inter s u) x
            hx : Membership.mem s x
            e : PartialHomeomorph H H
            heG : Membership.mem G e
            hef : Set.EqOn f (↑e.toPartialEquiv) (Inter.inter (Inter.inter s u) e.source)
            hex : Membership.mem e.source x
            ⊢ Membership.mem G (e.restr (interior u))
          -/
        · exact closedUnderRestriction' heG isOpen_interior
          /-
            🎉 no goals
          -/
          /-
            case mpr.intro.intro.intro.refine_2
            H : Type u_1
            inst✝¹ : TopologicalSpace H
            G : StructureGroupoid H
            inst✝ : ClosedUnderRestriction G
            s : Set H
            x : H
            u : Set H
            f : H → H
            hu : IsOpen u
            hux : Membership.mem u x
            h : G.IsLocalStructomorphWithinAt f (Inter.inter s u) x
            hx : Membership.mem s x
            e : PartialHomeomorph H H
            heG : Membership.mem G e
            hef : Set.EqOn f (↑e.toPartialEquiv) (Inter.inter (Inter.inter s u) e.source)
            hex : Membership.mem e.source x
            ⊢ Set.EqOn f (↑(e.restr (interior u)).toPartialEquiv) (Inter.inter s (e.restr  …
          -/
        · have : s ∩ u ∩ e.source = s ∩ (e.source ∩ u) := by mfld_set_tac
          /-
            case mpr.intro.intro.intro.refine_2
            H : Type u_1
            inst✝¹ : TopologicalSpace H
            G : StructureGroupoid H
            inst✝ : ClosedUnderRestriction G
            s : Set H
            x : H
            u : Set H
            f : H → H
            hu : IsOpen u
            hux : Membership.mem u x
            h : G.IsLocalStructomorphWithinAt f (Inter.inter s u) x
            hx : Membership.mem s x
            e : PartialHomeomorph H H
            heG : Membership.mem G e
            hef : Set.EqOn f (↑e.toPartialEquiv) (Inter.inter (Inter.inter s u) e.source)
            hex : Membership.mem e.source x
            this : Eq (Inter.inter (Inter.inter s u) e.source) (Inter.inter s (Inter.inter …
            ⊢ Set.EqOn f (↑(e.restr (interior u)).toPartialEquiv) (Inter.inter s (e.restr  …
          -/
          simpa only [this, interior_interior, hu.interior_eq, mfld_simps] using hef
          /-
            🎉 no goals
          -/
          /-
            case mpr.intro.intro.intro.refine_3
            H : Type u_1
            inst✝¹ : TopologicalSpace H
            G : StructureGroupoid H
            inst✝ : ClosedUnderRestriction G
            s : Set H
            x : H
            u : Set H
            f : H → H
            hu : IsOpen u
            hux : Membership.mem u x
            h : G.IsLocalStructomorphWithinAt f (Inter.inter s u) x
            hx : Membership.mem s x
            e : PartialHomeomorph H H
            heG : Membership.mem G e
            hef : Set.EqOn f (↑e.toPartialEquiv) (Inter.inter (Inter.inter s u) e.source)
            hex : Membership.mem e.source x
            ⊢ Membership.mem (e.restr (interior u)).source x
          -/
        · simp only [*, interior_interior, hu.interior_eq, mfld_simps]
          /-
            🎉 no goals
          -/
    right_invariance' := by
      /-
        H : Type u_1
        inst✝¹ : TopologicalSpace H
        G : StructureGroupoid H
        inst✝ : ClosedUnderRestriction G
        ⊢ ∀ {s : Set H} {x : H} {f : H → H} {e : PartialHomeomorph H H}, Membership.me …
      -/
      intro s x f e' he'G he'x h hx
      /-
        H : Type u_1
        inst✝¹ : TopologicalSpace H
        G : StructureGroupoid H
        inst✝ : ClosedUnderRestriction G
        s : Set H
        x : H
        f : H → H
        e' : PartialHomeomorph H H
        he'G : Membership.mem G e'
        he'x : Membership.mem e'.source x
        h : G.IsLocalStructomorphWithinAt f s x
        hx : Membership.mem (Set.preimage (↑e'.symm) s) (↑e' x)
        ⊢ Exists fun e => And (Membership.mem G e) (And (Set.EqOn (Function.comp f ↑e' …
      -/
      have hxs : x ∈ s := by simpa only [e'.left_inv he'x, mfld_simps] using hx
      /-
        H : Type u_1
        inst✝¹ : TopologicalSpace H
        G : StructureGroupoid H
        inst✝ : ClosedUnderRestriction G
        s : Set H
        x : H
        f : H → H
        e' : PartialHomeomorph H H
        he'G : Membership.mem G e'
        he'x : Membership.mem e'.source x
        h : G.IsLocalStructomorphWithinAt f s x
        hx : Membership.mem (Set.preimage (↑e'.symm) s) (↑e' x)
        hxs : Membership.mem s x
        ⊢ Exists fun e => And (Membership.mem G e) (And (Set.EqOn (Function.comp f ↑e' …
      -/
      rcases h hxs with ⟨e, heG, hef, hex⟩
      /-
        case intro.intro.intro
        H : Type u_1
        inst✝¹ : TopologicalSpace H
        G : StructureGroupoid H
        inst✝ : ClosedUnderRestriction G
        s : Set H
        x : H
        f : H → H
        e' : PartialHomeomorph H H
        he'G : Membership.mem G e'
        he'x : Membership.mem e'.source x
        h : G.IsLocalStructomorphWithinAt f s x
        hx : Membership.mem (Set.preimage (↑e'.symm) s) (↑e' x)
        hxs : Membership.mem s x
        e : PartialHomeomorph H H
        heG : Membership.mem G e
        hef : Set.EqOn f (↑e.toPartialEquiv) (Inter.inter s e.source)
        hex : Membership.mem e.source x
        ⊢ Exists fun e => And (Membership.mem G e) (And (Set.EqOn (Function.comp f ↑e' …
      -/
      refine ⟨e'.symm.trans e, G.trans (G.symm he'G) heG, ?_, ?_⟩
        /-
          case intro.intro.intro.refine_1
          H : Type u_1
          inst✝¹ : TopologicalSpace H
          G : StructureGroupoid H
          inst✝ : ClosedUnderRestriction G
          s : Set H
          x : H
          f : H → H
          e' : PartialHomeomorph H H
          he'G : Membership.mem G e'
          he'x : Membership.mem e'.source x
          h : G.IsLocalStructomorphWithinAt f s x
          hx : Membership.mem (Set.preimage (↑e'.symm) s) (↑e' x)
          hxs : Membership.mem s x
          e : PartialHomeomorph H H
          heG : Membership.mem G e
          hef : Set.EqOn f (↑e.toPartialEquiv) (Inter.inter s e.source)
          hex : Membership.mem e.source x
          ⊢ Set.EqOn (Function.comp f ↑e'.symm) (↑(e'.symm.trans e).toPartialEquiv) (Int …
        -/
      · intro y hy
        /-
          case intro.intro.intro.refine_1
          H : Type u_1
          inst✝¹ : TopologicalSpace H
          G : StructureGroupoid H
          inst✝ : ClosedUnderRestriction G
          s : Set H
          x : H
          f : H → H
          e' : PartialHomeomorph H H
          he'G : Membership.mem G e'
          he'x : Membership.mem e'.source x
          h : G.IsLocalStructomorphWithinAt f s x
          hx : Membership.mem (Set.preimage (↑e'.symm) s) (↑e' x)
          hxs : Membership.mem s x
          e : PartialHomeomorph H H
          heG : Membership.mem G e
          hef : Set.EqOn f (↑e.toPartialEquiv) (Inter.inter s e.source)
          hex : Membership.mem e.source x
          y : H
          hy : Membership.mem (Inter.inter (Set.preimage (↑e'.symm) s) (e'.symm.trans e) …
          ⊢ Eq (Function.comp f (↑e'.symm) y) (↑(e'.symm.trans e).toPartialEquiv y)
        -/
        simp only [mfld_simps] at hy
        /-
          case intro.intro.intro.refine_1
          H : Type u_1
          inst✝¹ : TopologicalSpace H
          G : StructureGroupoid H
          inst✝ : ClosedUnderRestriction G
          s : Set H
          x : H
          f : H → H
          e' : PartialHomeomorph H H
          he'G : Membership.mem G e'
          he'x : Membership.mem e'.source x
          h : G.IsLocalStructomorphWithinAt f s x
          hx : Membership.mem (Set.preimage (↑e'.symm) s) (↑e' x)
          hxs : Membership.mem s x
          e : PartialHomeomorph H H
          heG : Membership.mem G e
          hef : Set.EqOn f (↑e.toPartialEquiv) (Inter.inter s e.source)
          hex : Membership.mem e.source x
          y : H
          hy : And (Membership.mem s (↑e'.symm y)) (And (Membership.mem e'.target y) (Me …
          ⊢ Eq (Function.comp f (↑e'.symm) y) (↑(e'.symm.trans e).toPartialEquiv y)
        -/
        simp only [hef ⟨hy.1, hy.2.2⟩, mfld_simps]
        /-
          🎉 no goals
        -/
        /-
          case intro.intro.intro.refine_2
          H : Type u_1
          inst✝¹ : TopologicalSpace H
          G : StructureGroupoid H
          inst✝ : ClosedUnderRestriction G
          s : Set H
          x : H
          f : H → H
          e' : PartialHomeomorph H H
          he'G : Membership.mem G e'
          he'x : Membership.mem e'.source x
          h : G.IsLocalStructomorphWithinAt f s x
          hx : Membership.mem (Set.preimage (↑e'.symm) s) (↑e' x)
          hxs : Membership.mem s x
          e : PartialHomeomorph H H
          heG : Membership.mem G e
          hef : Set.EqOn f (↑e.toPartialEquiv) (Inter.inter s e.source)
          hex : Membership.mem e.source x
          ⊢ Membership.mem (e'.symm.trans e).source (↑e' x)
        -/
      · simp only [hex, he'x, mfld_simps]
        /-
          🎉 no goals
        -/
    congr_of_forall := by
      /-
        H : Type u_1
        inst✝¹ : TopologicalSpace H
        G : StructureGroupoid H
        inst✝ : ClosedUnderRestriction G
        ⊢ ∀ {s : Set H} {x : H} {f g : H → H}, (∀ (y : H), Membership.mem s y → Eq (f  …
      -/
      intro s x f g hfgs _ h hx
      /-
        H : Type u_1
        inst✝¹ : TopologicalSpace H
        G : StructureGroupoid H
        inst✝ : ClosedUnderRestriction G
        s : Set H
        x : H
        f g : H → H
        hfgs : ∀ (y : H), Membership.mem s y → Eq (f y) (g y)
        a✝ : Eq (f x) (g x)
        h : G.IsLocalStructomorphWithinAt f s x
        hx : Membership.mem s x
        ⊢ Exists fun e => And (Membership.mem G e) (And (Set.EqOn g (↑e.toPartialEquiv …
      -/
      rcases h hx with ⟨e, heG, hef, hex⟩
      /-
        case intro.intro.intro
        H : Type u_1
        inst✝¹ : TopologicalSpace H
        G : StructureGroupoid H
        inst✝ : ClosedUnderRestriction G
        s : Set H
        x : H
        f g : H → H
        hfgs : ∀ (y : H), Membership.mem s y → Eq (f y) (g y)
        a✝ : Eq (f x) (g x)
        h : G.IsLocalStructomorphWithinAt f s x
        hx : Membership.mem s x
        e : PartialHomeomorph H H
        heG : Membership.mem G e
        hef : Set.EqOn f (↑e.toPartialEquiv) (Inter.inter s e.source)
        hex : Membership.mem e.source x
        ⊢ Exists fun e => And (Membership.mem G e) (And (Set.EqOn g (↑e.toPartialEquiv …
      -/
      refine ⟨e, heG, ?_, hex⟩
      /-
        case intro.intro.intro
        H : Type u_1
        inst✝¹ : TopologicalSpace H
        G : StructureGroupoid H
        inst✝ : ClosedUnderRestriction G
        s : Set H
        x : H
        f g : H → H
        hfgs : ∀ (y : H), Membership.mem s y → Eq (f y) (g y)
        a✝ : Eq (f x) (g x)
        h : G.IsLocalStructomorphWithinAt f s x
        hx : Membership.mem s x
        e : PartialHomeomorph H H
        heG : Membership.mem G e
        hef : Set.EqOn f (↑e.toPartialEquiv) (Inter.inter s e.source)
        hex : Membership.mem e.source x
        ⊢ Set.EqOn g (↑e.toPartialEquiv) (Inter.inter s e.source)
      -/
      intro y hy
      /-
        case intro.intro.intro
        H : Type u_1
        inst✝¹ : TopologicalSpace H
        G : StructureGroupoid H
        inst✝ : ClosedUnderRestriction G
        s : Set H
        x : H
        f g : H → H
        hfgs : ∀ (y : H), Membership.mem s y → Eq (f y) (g y)
        a✝ : Eq (f x) (g x)
        h : G.IsLocalStructomorphWithinAt f s x
        hx : Membership.mem s x
        e : PartialHomeomorph H H
        heG : Membership.mem G e
        hef : Set.EqOn f (↑e.toPartialEquiv) (Inter.inter s e.source)
        hex : Membership.mem e.source x
        y : H
        hy : Membership.mem (Inter.inter s e.source) y
        ⊢ Eq (g y) (↑e.toPartialEquiv y)
      -/
      rw [← hef hy, hfgs y hy.1]
      /-
        🎉 no goals
      -/
    left_invariance' := by
      /-
        H : Type u_1
        inst✝¹ : TopologicalSpace H
        G : StructureGroupoid H
        inst✝ : ClosedUnderRestriction G
        ⊢ ∀ {s : Set H} {x : H} {f : H → H} {e' : PartialHomeomorph H H}, Membership.m …
      -/
      intro s x f e' he'G _ hfx h hx
      /-
        H : Type u_1
        inst✝¹ : TopologicalSpace H
        G : StructureGroupoid H
        inst✝ : ClosedUnderRestriction G
        s : Set H
        x : H
        f : H → H
        e' : PartialHomeomorph H H
        he'G : Membership.mem G e'
        a✝ : HasSubset.Subset s (Set.preimage f e'.source)
        hfx : Membership.mem e'.source (f x)
        h : G.IsLocalStructomorphWithinAt f s x
        hx : Membership.mem s x
        ⊢ Exists fun e => And (Membership.mem G e) (And (Set.EqOn (Function.comp (↑e') …
      -/
      rcases h hx with ⟨e, heG, hef, hex⟩
      /-
        case intro.intro.intro
        H : Type u_1
        inst✝¹ : TopologicalSpace H
        G : StructureGroupoid H
        inst✝ : ClosedUnderRestriction G
        s : Set H
        x : H
        f : H → H
        e' : PartialHomeomorph H H
        he'G : Membership.mem G e'
        a✝ : HasSubset.Subset s (Set.preimage f e'.source)
        hfx : Membership.mem e'.source (f x)
        h : G.IsLocalStructomorphWithinAt f s x
        hx : Membership.mem s x
        e : PartialHomeomorph H H
        heG : Membership.mem G e
        hef : Set.EqOn f (↑e.toPartialEquiv) (Inter.inter s e.source)
        hex : Membership.mem e.source x
        ⊢ Exists fun e => And (Membership.mem G e) (And (Set.EqOn (Function.comp (↑e') …
      -/
      refine ⟨e.trans e', G.trans heG he'G, ?_, ?_⟩
        /-
          case intro.intro.intro.refine_1
          H : Type u_1
          inst✝¹ : TopologicalSpace H
          G : StructureGroupoid H
          inst✝ : ClosedUnderRestriction G
          s : Set H
          x : H
          f : H → H
          e' : PartialHomeomorph H H
          he'G : Membership.mem G e'
          a✝ : HasSubset.Subset s (Set.preimage f e'.source)
          hfx : Membership.mem e'.source (f x)
          h : G.IsLocalStructomorphWithinAt f s x
          hx : Membership.mem s x
          e : PartialHomeomorph H H
          heG : Membership.mem G e
          hef : Set.EqOn f (↑e.toPartialEquiv) (Inter.inter s e.source)
          hex : Membership.mem e.source x
          ⊢ Set.EqOn (Function.comp (↑e') f) (↑(e.trans e').toPartialEquiv) (Inter.inter …
        -/
      · intro y hy
        /-
          case intro.intro.intro.refine_1
          H : Type u_1
          inst✝¹ : TopologicalSpace H
          G : StructureGroupoid H
          inst✝ : ClosedUnderRestriction G
          s : Set H
          x : H
          f : H → H
          e' : PartialHomeomorph H H
          he'G : Membership.mem G e'
          a✝ : HasSubset.Subset s (Set.preimage f e'.source)
          hfx : Membership.mem e'.source (f x)
          h : G.IsLocalStructomorphWithinAt f s x
          hx : Membership.mem s x
          e : PartialHomeomorph H H
          heG : Membership.mem G e
          hef : Set.EqOn f (↑e.toPartialEquiv) (Inter.inter s e.source)
          hex : Membership.mem e.source x
          y : H
          hy : Membership.mem (Inter.inter s (e.trans e').source) y
          ⊢ Eq (Function.comp (↑e') f y) (↑(e.trans e').toPartialEquiv y)
        -/
        simp only [mfld_simps] at hy
        /-
          case intro.intro.intro.refine_1
          H : Type u_1
          inst✝¹ : TopologicalSpace H
          G : StructureGroupoid H
          inst✝ : ClosedUnderRestriction G
          s : Set H
          x : H
          f : H → H
          e' : PartialHomeomorph H H
          he'G : Membership.mem G e'
          a✝ : HasSubset.Subset s (Set.preimage f e'.source)
          hfx : Membership.mem e'.source (f x)
          h : G.IsLocalStructomorphWithinAt f s x
          hx : Membership.mem s x
          e : PartialHomeomorph H H
          heG : Membership.mem G e
          hef : Set.EqOn f (↑e.toPartialEquiv) (Inter.inter s e.source)
          hex : Membership.mem e.source x
          y : H
          hy : And (Membership.mem s y) (And (Membership.mem e.source y) (Membership.mem …
          ⊢ Eq (Function.comp (↑e') f y) (↑(e.trans e').toPartialEquiv y)
        -/
        simp only [hef ⟨hy.1, hy.2.1⟩, mfld_simps]
        /-
          🎉 no goals
        -/
        /-
          case intro.intro.intro.refine_2
          H : Type u_1
          inst✝¹ : TopologicalSpace H
          G : StructureGroupoid H
          inst✝ : ClosedUnderRestriction G
          s : Set H
          x : H
          f : H → H
          e' : PartialHomeomorph H H
          he'G : Membership.mem G e'
          a✝ : HasSubset.Subset s (Set.preimage f e'.source)
          hfx : Membership.mem e'.source (f x)
          h : G.IsLocalStructomorphWithinAt f s x
          hx : Membership.mem s x
          e : PartialHomeomorph H H
          heG : Membership.mem G e
          hef : Set.EqOn f (↑e.toPartialEquiv) (Inter.inter s e.source)
          hex : Membership.mem e.source x
          ⊢ Membership.mem (e.trans e').source x
        -/
      · simpa only [hex, hef ⟨hx, hex⟩, mfld_simps] using hfx }
        /-
          🎉 no goals
        -/


/-- A slight reformulation of `IsLocalStructomorphWithinAt` when `f` is a partial homeomorph.
  This gives us an `e` that is defined on a subset of `f.source`. -/
theorem _root_.PartialHomeomorph.isLocalStructomorphWithinAt_iff {G : StructureGroupoid H}
    [ClosedUnderRestriction G] (f : PartialHomeomorph H H) {s : Set H} {x : H}
    (hx : x ∈ f.source ∪ sᶜ) :
    G.IsLocalStructomorphWithinAt (⇑f) s x ↔
      x ∈ s → ∃ e : PartialHomeomorph H H,
      e ∈ G ∧ e.source ⊆ f.source ∧ EqOn f (⇑e) (s ∩ e.source) ∧ x ∈ e.source := by
  /-
    H : Type u_1
    inst✝¹ : TopologicalSpace H
    G : StructureGroupoid H
    inst✝ : ClosedUnderRestriction G
    f : PartialHomeomorph H H
    s : Set H
    x : H
    hx : Membership.mem (Union.union f.source (HasCompl.compl s)) x
    ⊢ Iff (G.IsLocalStructomorphWithinAt (↑f) s x) (Membership.mem s x → Exists fu …
  -/
  constructor
    /-
      case mp
      H : Type u_1
      inst✝¹ : TopologicalSpace H
      G : StructureGroupoid H
      inst✝ : ClosedUnderRestriction G
      f : PartialHomeomorph H H
      s : Set H
      x : H
      hx : Membership.mem (Union.union f.source (HasCompl.compl s)) x
      ⊢ G.IsLocalStructomorphWithinAt (↑f) s x → Membership.mem s x → Exists fun e = …
    -/
  · intro hf h2x
    /-
      case mp
      H : Type u_1
      inst✝¹ : TopologicalSpace H
      G : StructureGroupoid H
      inst✝ : ClosedUnderRestriction G
      f : PartialHomeomorph H H
      s : Set H
      x : H
      hx : Membership.mem (Union.union f.source (HasCompl.compl s)) x
      hf : G.IsLocalStructomorphWithinAt (↑f) s x
      h2x : Membership.mem s x
      ⊢ Exists fun e => And (Membership.mem G e) (And (HasSubset.Subset e.source f.s …
    -/
    obtain ⟨e, he, hfe, hxe⟩ := hf h2x
    /-
      case mp.intro.intro.intro
      H : Type u_1
      inst✝¹ : TopologicalSpace H
      G : StructureGroupoid H
      inst✝ : ClosedUnderRestriction G
      f : PartialHomeomorph H H
      s : Set H
      x : H
      hx : Membership.mem (Union.union f.source (HasCompl.compl s)) x
      hf : G.IsLocalStructomorphWithinAt (↑f) s x
      h2x : Membership.mem s x
      e : PartialHomeomorph H H
      he : Membership.mem G e
      hfe : Set.EqOn (↑f) (↑e.toPartialEquiv) (Inter.inter s e.source)
      hxe : Membership.mem e.source x
      ⊢ Exists fun e => And (Membership.mem G e) (And (HasSubset.Subset e.source f.s …
    -/
    refine ⟨e.restr f.source, closedUnderRestriction' he f.open_source, ?_, ?_, hxe, ?_⟩
      /-
        case mp.intro.intro.intro.refine_1
        H : Type u_1
        inst✝¹ : TopologicalSpace H
        G : StructureGroupoid H
        inst✝ : ClosedUnderRestriction G
        f : PartialHomeomorph H H
        s : Set H
        x : H
        hx : Membership.mem (Union.union f.source (HasCompl.compl s)) x
        hf : G.IsLocalStructomorphWithinAt (↑f) s x
        h2x : Membership.mem s x
        e : PartialHomeomorph H H
        he : Membership.mem G e
        hfe : Set.EqOn (↑f) (↑e.toPartialEquiv) (Inter.inter s e.source)
        hxe : Membership.mem e.source x
        ⊢ HasSubset.Subset (e.restr f.source).source f.source
      -/
    · simp_rw [PartialHomeomorph.restr_source]
      /-
        case mp.intro.intro.intro.refine_1
        H : Type u_1
        inst✝¹ : TopologicalSpace H
        G : StructureGroupoid H
        inst✝ : ClosedUnderRestriction G
        f : PartialHomeomorph H H
        s : Set H
        x : H
        hx : Membership.mem (Union.union f.source (HasCompl.compl s)) x
        hf : G.IsLocalStructomorphWithinAt (↑f) s x
        h2x : Membership.mem s x
        e : PartialHomeomorph H H
        he : Membership.mem G e
        hfe : Set.EqOn (↑f) (↑e.toPartialEquiv) (Inter.inter s e.source)
        hxe : Membership.mem e.source x
        ⊢ HasSubset.Subset (Inter.inter e.source (interior f.source)) f.source
      -/
      exact inter_subset_right.trans interior_subset
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.intro.refine_2
        H : Type u_1
        inst✝¹ : TopologicalSpace H
        G : StructureGroupoid H
        inst✝ : ClosedUnderRestriction G
        f : PartialHomeomorph H H
        s : Set H
        x : H
        hx : Membership.mem (Union.union f.source (HasCompl.compl s)) x
        hf : G.IsLocalStructomorphWithinAt (↑f) s x
        h2x : Membership.mem s x
        e : PartialHomeomorph H H
        he : Membership.mem G e
        hfe : Set.EqOn (↑f) (↑e.toPartialEquiv) (Inter.inter s e.source)
        hxe : Membership.mem e.source x
        ⊢ Set.EqOn (↑f) (↑(e.restr f.source)) (Inter.inter s (e.restr f.source).source)
      -/
    · intro x' hx'
      /-
        case mp.intro.intro.intro.refine_2
        H : Type u_1
        inst✝¹ : TopologicalSpace H
        G : StructureGroupoid H
        inst✝ : ClosedUnderRestriction G
        f : PartialHomeomorph H H
        s : Set H
        x : H
        hx : Membership.mem (Union.union f.source (HasCompl.compl s)) x
        hf : G.IsLocalStructomorphWithinAt (↑f) s x
        h2x : Membership.mem s x
        e : PartialHomeomorph H H
        he : Membership.mem G e
        hfe : Set.EqOn (↑f) (↑e.toPartialEquiv) (Inter.inter s e.source)
        hxe : Membership.mem e.source x
        x' : H
        hx' : Membership.mem (Inter.inter s (e.restr f.source).source) x'
        ⊢ Eq (↑f x') (↑(e.restr f.source) x')
      -/
      exact hfe ⟨hx'.1, hx'.2.1⟩
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.intro.refine_3
        H : Type u_1
        inst✝¹ : TopologicalSpace H
        G : StructureGroupoid H
        inst✝ : ClosedUnderRestriction G
        f : PartialHomeomorph H H
        s : Set H
        x : H
        hx : Membership.mem (Union.union f.source (HasCompl.compl s)) x
        hf : G.IsLocalStructomorphWithinAt (↑f) s x
        h2x : Membership.mem s x
        e : PartialHomeomorph H H
        he : Membership.mem G e
        hfe : Set.EqOn (↑f) (↑e.toPartialEquiv) (Inter.inter s e.source)
        hxe : Membership.mem e.source x
        ⊢ Membership.mem (interior f.source) x
      -/
    · rw [f.open_source.interior_eq]
      /-
        case mp.intro.intro.intro.refine_3
        H : Type u_1
        inst✝¹ : TopologicalSpace H
        G : StructureGroupoid H
        inst✝ : ClosedUnderRestriction G
        f : PartialHomeomorph H H
        s : Set H
        x : H
        hx : Membership.mem (Union.union f.source (HasCompl.compl s)) x
        hf : G.IsLocalStructomorphWithinAt (↑f) s x
        h2x : Membership.mem s x
        e : PartialHomeomorph H H
        he : Membership.mem G e
        hfe : Set.EqOn (↑f) (↑e.toPartialEquiv) (Inter.inter s e.source)
        hxe : Membership.mem e.source x
        ⊢ Membership.mem f.source x
      -/
      exact Or.resolve_right hx (not_not.mpr h2x)
      /-
        🎉 no goals
      -/
    /-
      case mpr
      H : Type u_1
      inst✝¹ : TopologicalSpace H
      G : StructureGroupoid H
      inst✝ : ClosedUnderRestriction G
      f : PartialHomeomorph H H
      s : Set H
      x : H
      hx : Membership.mem (Union.union f.source (HasCompl.compl s)) x
      ⊢ (Membership.mem s x → Exists fun e => And (Membership.mem G e) (And (HasSubs …
    -/
  · intro hf hx
    /-
      case mpr
      H : Type u_1
      inst✝¹ : TopologicalSpace H
      G : StructureGroupoid H
      inst✝ : ClosedUnderRestriction G
      f : PartialHomeomorph H H
      s : Set H
      x : H
      hx✝ : Membership.mem (Union.union f.source (HasCompl.compl s)) x
      hf : Membership.mem s x → Exists fun e => And (Membership.mem G e) (And (HasSu …
      hx : Membership.mem s x
      ⊢ Exists fun e => And (Membership.mem G e) (And (Set.EqOn (↑f) (↑e.toPartialEq …
    -/
    obtain ⟨e, he, _, hfe, hxe⟩ := hf hx
    /-
      case mpr.intro.intro.intro.intro
      H : Type u_1
      inst✝¹ : TopologicalSpace H
      G : StructureGroupoid H
      inst✝ : ClosedUnderRestriction G
      f : PartialHomeomorph H H
      s : Set H
      x : H
      hx✝ : Membership.mem (Union.union f.source (HasCompl.compl s)) x
      hf : Membership.mem s x → Exists fun e => And (Membership.mem G e) (And (HasSu …
      hx : Membership.mem s x
      e : PartialHomeomorph H H
      he : Membership.mem G e
      left✝ : HasSubset.Subset e.source f.source
      hfe : Set.EqOn (↑f) (↑e) (Inter.inter s e.source)
      hxe : Membership.mem e.source x
      ⊢ Exists fun e => And (Membership.mem G e) (And (Set.EqOn (↑f) (↑e.toPartialEq …
    -/
    exact ⟨e, he, hfe, hxe⟩
    /-
      🎉 no goals
    -/


/-- A slight reformulation of `IsLocalStructomorphWithinAt` when `f` is a partial homeomorph and
  the set we're considering is a superset of `f.source`. -/
theorem _root_.PartialHomeomorph.isLocalStructomorphWithinAt_iff' {G : StructureGroupoid H}
    [ClosedUnderRestriction G] (f : PartialHomeomorph H H) {s : Set H} {x : H} (hs : f.source ⊆ s)
    (hx : x ∈ f.source ∪ sᶜ) :
    G.IsLocalStructomorphWithinAt (⇑f) s x ↔
      x ∈ s → ∃ e : PartialHomeomorph H H,
      e ∈ G ∧ e.source ⊆ f.source ∧ EqOn f (⇑e) e.source ∧ x ∈ e.source := by
  /-
    H : Type u_1
    inst✝¹ : TopologicalSpace H
    G : StructureGroupoid H
    inst✝ : ClosedUnderRestriction G
    f : PartialHomeomorph H H
    s : Set H
    x : H
    hs : HasSubset.Subset f.source s
    hx : Membership.mem (Union.union f.source (HasCompl.compl s)) x
    ⊢ Iff (G.IsLocalStructomorphWithinAt (↑f) s x) (Membership.mem s x → Exists fu …
  -/
  rw [f.isLocalStructomorphWithinAt_iff hx]
  /-
    H : Type u_1
    inst✝¹ : TopologicalSpace H
    G : StructureGroupoid H
    inst✝ : ClosedUnderRestriction G
    f : PartialHomeomorph H H
    s : Set H
    x : H
    hs : HasSubset.Subset f.source s
    hx : Membership.mem (Union.union f.source (HasCompl.compl s)) x
    ⊢ Iff (Membership.mem s x → Exists fun e => And (Membership.mem G e) (And (Has …
  -/
  refine imp_congr_right fun _ ↦ exists_congr fun e ↦ and_congr_right fun _ ↦ ?_
  /-
    H : Type u_1
    inst✝¹ : TopologicalSpace H
    G : StructureGroupoid H
    inst✝ : ClosedUnderRestriction G
    f : PartialHomeomorph H H
    s : Set H
    x : H
    hs : HasSubset.Subset f.source s
    hx : Membership.mem (Union.union f.source (HasCompl.compl s)) x
    x✝¹ : Membership.mem s x
    e : PartialHomeomorph H H
    x✝ : Membership.mem G e
    ⊢ Iff (And (HasSubset.Subset e.source f.source) (And (Set.EqOn (↑f) (↑e) (Inte …
  -/
  refine and_congr_right fun h2e ↦ ?_
  /-
    H : Type u_1
    inst✝¹ : TopologicalSpace H
    G : StructureGroupoid H
    inst✝ : ClosedUnderRestriction G
    f : PartialHomeomorph H H
    s : Set H
    x : H
    hs : HasSubset.Subset f.source s
    hx : Membership.mem (Union.union f.source (HasCompl.compl s)) x
    x✝¹ : Membership.mem s x
    e : PartialHomeomorph H H
    x✝ : Membership.mem G e
    h2e : HasSubset.Subset e.source f.source
    ⊢ Iff (And (Set.EqOn (↑f) (↑e) (Inter.inter s e.source)) (Membership.mem e.sou …
  -/
  rw [inter_eq_right.mpr (h2e.trans hs)]
  /-
    🎉 no goals
  -/


/-- A slight reformulation of `IsLocalStructomorphWithinAt` when `f` is a partial homeomorph and
  the set we're considering is `f.source`. -/
theorem _root_.PartialHomeomorph.isLocalStructomorphWithinAt_source_iff {G : StructureGroupoid H}
    [ClosedUnderRestriction G] (f : PartialHomeomorph H H) {x : H} :
    G.IsLocalStructomorphWithinAt (⇑f) f.source x ↔
      x ∈ f.source → ∃ e : PartialHomeomorph H H,
      e ∈ G ∧ e.source ⊆ f.source ∧ EqOn f (⇑e) e.source ∧ x ∈ e.source :=
                                         /-
                                           H : Type u_1
                                           inst✝¹ : TopologicalSpace H
                                           G : StructureGroupoid H
                                           inst✝ : ClosedUnderRestriction G
                                           f : PartialHomeomorph H H
                                           x : H
                                           ⊢ Membership.mem (Union.union f.source (HasCompl.compl f.source)) x
                                         -/
  haveI : x ∈ f.source ∪ f.sourceᶜ := by simp_rw [union_compl_self, mem_univ]
                                         /-
                                           🎉 no goals
                                         -/
  f.isLocalStructomorphWithinAt_iff' Subset.rfl this


theorem HasGroupoid.comp
    (H : ∀ e ∈ G₂, LiftPropOn (IsLocalStructomorphWithinAt G₁) (e : H₂ → H₂) e.source) :
    @HasGroupoid H₁ _ H₃ _ (ChartedSpace.comp H₁ H₂ H₃) G₁ :=
  let _ := ChartedSpace.comp H₁ H₂ H₃ -- Porting note: need this to synthesize `ChartedSpace H₁ H₃`
  { compatible := by
      /-
        H₁ : Type u_6
        inst✝⁷ : TopologicalSpace H₁
        H₂ : Type u_7
        inst✝⁶ : TopologicalSpace H₂
        H₃ : Type u_8
        inst✝⁵ : TopologicalSpace H₃
        inst✝⁴ : ChartedSpace H₁ H₂
        inst✝³ : ChartedSpace H₂ H₃
        G₁ : StructureGroupoid H₁
        inst✝² : HasGroupoid H₂ G₁
        inst✝¹ : ClosedUnderRestriction G₁
        G₂ : StructureGroupoid H₂
        inst✝ : HasGroupoid H₃ G₂
        H : ∀ (e : PartialHomeomorph H₂ H₂), Membership.mem G₂ e → ChartedSpace.LiftPr …
        x✝ : ChartedSpace H₁ H₃ := ChartedSpace.comp H₁ H₂ H₃
        ⊢ ∀ {e e' : PartialHomeomorph H₃ H₁}, Membership.mem (atlas H₁ H₃) e → Members …
      -/
      rintro _ _ ⟨e, he, f, hf, rfl⟩ ⟨e', he', f', hf', rfl⟩
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro
        H₁ : Type u_6
        inst✝⁷ : TopologicalSpace H₁
        H₂ : Type u_7
        inst✝⁶ : TopologicalSpace H₂
        H₃ : Type u_8
        inst✝⁵ : TopologicalSpace H₃
        inst✝⁴ : ChartedSpace H₁ H₂
        inst✝³ : ChartedSpace H₂ H₃
        G₁ : StructureGroupoid H₁
        inst✝² : HasGroupoid H₂ G₁
        inst✝¹ : ClosedUnderRestriction G₁
        G₂ : StructureGroupoid H₂
        inst✝ : HasGroupoid H₃ G₂
        H : ∀ (e : PartialHomeomorph H₂ H₂), Membership.mem G₂ e → ChartedSpace.LiftPr …
        x✝ : ChartedSpace H₁ H₃ := ChartedSpace.comp H₁ H₂ H₃
        e : PartialHomeomorph H₃ H₂
        he : Membership.mem (atlas H₂ H₃) e
        f : PartialHomeomorph H₂ H₁
        hf : Membership.mem (atlas H₁ H₂) f
        e' : PartialHomeomorph H₃ H₂
        he' : Membership.mem (atlas H₂ H₃) e'
        f' : PartialHomeomorph H₂ H₁
        hf' : Membership.mem (atlas H₁ H₂) f'
        ⊢ Membership.mem G₁ ((e.trans f).symm.trans (e'.trans f'))
      -/
      apply G₁.locality
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro
        H₁ : Type u_6
        inst✝⁷ : TopologicalSpace H₁
        H₂ : Type u_7
        inst✝⁶ : TopologicalSpace H₂
        H₃ : Type u_8
        inst✝⁵ : TopologicalSpace H₃
        inst✝⁴ : ChartedSpace H₁ H₂
        inst✝³ : ChartedSpace H₂ H₃
        G₁ : StructureGroupoid H₁
        inst✝² : HasGroupoid H₂ G₁
        inst✝¹ : ClosedUnderRestriction G₁
        G₂ : StructureGroupoid H₂
        inst✝ : HasGroupoid H₃ G₂
        H : ∀ (e : PartialHomeomorph H₂ H₂), Membership.mem G₂ e → ChartedSpace.LiftPr …
        x✝ : ChartedSpace H₁ H₃ := ChartedSpace.comp H₁ H₂ H₃
        e : PartialHomeomorph H₃ H₂
        he : Membership.mem (atlas H₂ H₃) e
        f : PartialHomeomorph H₂ H₁
        hf : Membership.mem (atlas H₁ H₂) f
        e' : PartialHomeomorph H₃ H₂
        he' : Membership.mem (atlas H₂ H₃) e'
        f' : PartialHomeomorph H₂ H₁
        hf' : Membership.mem (atlas H₁ H₂) f'
        ⊢ ∀ (x : H₁), Membership.mem ((e.trans f).symm.trans (e'.trans f')).source x → …
      -/
      intro x hx
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro
        H₁ : Type u_6
        inst✝⁷ : TopologicalSpace H₁
        H₂ : Type u_7
        inst✝⁶ : TopologicalSpace H₂
        H₃ : Type u_8
        inst✝⁵ : TopologicalSpace H₃
        inst✝⁴ : ChartedSpace H₁ H₂
        inst✝³ : ChartedSpace H₂ H₃
        G₁ : StructureGroupoid H₁
        inst✝² : HasGroupoid H₂ G₁
        inst✝¹ : ClosedUnderRestriction G₁
        G₂ : StructureGroupoid H₂
        inst✝ : HasGroupoid H₃ G₂
        H : ∀ (e : PartialHomeomorph H₂ H₂), Membership.mem G₂ e → ChartedSpace.LiftPr …
        x✝ : ChartedSpace H₁ H₃ := ChartedSpace.comp H₁ H₂ H₃
        e : PartialHomeomorph H₃ H₂
        he : Membership.mem (atlas H₂ H₃) e
        f : PartialHomeomorph H₂ H₁
        hf : Membership.mem (atlas H₁ H₂) f
        e' : PartialHomeomorph H₃ H₂
        he' : Membership.mem (atlas H₂ H₃) e'
        f' : PartialHomeomorph H₂ H₁
        hf' : Membership.mem (atlas H₁ H₂) f'
        x : H₁
        hx : Membership.mem ((e.trans f).symm.trans (e'.trans f')).source x
        ⊢ Exists fun s => And (IsOpen s) (And (Membership.mem s x) (Membership.mem G₁  …
      -/
      simp only [mfld_simps] at hx
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro
        H₁ : Type u_6
        inst✝⁷ : TopologicalSpace H₁
        H₂ : Type u_7
        inst✝⁶ : TopologicalSpace H₂
        H₃ : Type u_8
        inst✝⁵ : TopologicalSpace H₃
        inst✝⁴ : ChartedSpace H₁ H₂
        inst✝³ : ChartedSpace H₂ H₃
        G₁ : StructureGroupoid H₁
        inst✝² : HasGroupoid H₂ G₁
        inst✝¹ : ClosedUnderRestriction G₁
        G₂ : StructureGroupoid H₂
        inst✝ : HasGroupoid H₃ G₂
        H : ∀ (e : PartialHomeomorph H₂ H₂), Membership.mem G₂ e → ChartedSpace.LiftPr …
        x✝ : ChartedSpace H₁ H₃ := ChartedSpace.comp H₁ H₂ H₃
        e : PartialHomeomorph H₃ H₂
        he : Membership.mem (atlas H₂ H₃) e
        f : PartialHomeomorph H₂ H₁
        hf : Membership.mem (atlas H₁ H₂) f
        e' : PartialHomeomorph H₃ H₂
        he' : Membership.mem (atlas H₂ H₃) e'
        f' : PartialHomeomorph H₂ H₁
        hf' : Membership.mem (atlas H₁ H₂) f'
        x : H₁
        hx : And (And (Membership.mem f.target x) (Membership.mem e.target (↑f.symm x) …
        ⊢ Exists fun s => And (IsOpen s) (And (Membership.mem s x) (Membership.mem G₁  …
      -/
      have hxs : x ∈ f.symm ⁻¹' (e.symm ≫ₕ e').source := by simp only [hx, mfld_simps]
      have hxs' : x ∈ f.target ∩ f.symm ⁻¹' ((e.symm ≫ₕ e').source ∩ e.symm ≫ₕ e' ⁻¹' f'.source) :=
        by simp only [hx, mfld_simps]
      obtain ⟨φ, hφG₁, hφ, hφ_dom⟩ := LocalInvariantProp.liftPropOn_indep_chart
        (isLocalStructomorphWithinAt_localInvariantProp G₁) (G₁.subset_maximalAtlas hf)
        (G₁.subset_maximalAtlas hf') (H _ (G₂.compatible he he')) hxs' hxs
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
        H₁ : Type u_6
        inst✝⁷ : TopologicalSpace H₁
        H₂ : Type u_7
        inst✝⁶ : TopologicalSpace H₂
        H₃ : Type u_8
        inst✝⁵ : TopologicalSpace H₃
        inst✝⁴ : ChartedSpace H₁ H₂
        inst✝³ : ChartedSpace H₂ H₃
        G₁ : StructureGroupoid H₁
        inst✝² : HasGroupoid H₂ G₁
        inst✝¹ : ClosedUnderRestriction G₁
        G₂ : StructureGroupoid H₂
        inst✝ : HasGroupoid H₃ G₂
        H : ∀ (e : PartialHomeomorph H₂ H₂), Membership.mem G₂ e → ChartedSpace.LiftPr …
        x✝ : ChartedSpace H₁ H₃ := ChartedSpace.comp H₁ H₂ H₃
        e : PartialHomeomorph H₃ H₂
        he : Membership.mem (atlas H₂ H₃) e
        f : PartialHomeomorph H₂ H₁
        hf : Membership.mem (atlas H₁ H₂) f
        e' : PartialHomeomorph H₃ H₂
        he' : Membership.mem (atlas H₂ H₃) e'
        f' : PartialHomeomorph H₂ H₁
        hf' : Membership.mem (atlas H₁ H₂) f'
        x : H₁
        hx : And (And (Membership.mem f.target x) (Membership.mem e.target (↑f.symm x) …
        hxs : Membership.mem (Set.preimage (↑f.symm) (e.symm.trans e').source) x
        hxs' : Membership.mem (Inter.inter f.target (Set.preimage (↑f.symm) (Inter.int …
        φ : PartialHomeomorph H₁ H₁
        hφG₁ : Membership.mem G₁ φ
        hφ : Set.EqOn (Function.comp (↑f') (Function.comp ↑(e.symm.trans e') ↑f.symm)) …
        hφ_dom : Membership.mem φ.source x
        ⊢ Exists fun s => And (IsOpen s) (And (Membership.mem s x) (Membership.mem G₁  …
      -/
      simp_rw [← PartialHomeomorph.coe_trans, PartialHomeomorph.trans_assoc] at hφ
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
        H₁ : Type u_6
        inst✝⁷ : TopologicalSpace H₁
        H₂ : Type u_7
        inst✝⁶ : TopologicalSpace H₂
        H₃ : Type u_8
        inst✝⁵ : TopologicalSpace H₃
        inst✝⁴ : ChartedSpace H₁ H₂
        inst✝³ : ChartedSpace H₂ H₃
        G₁ : StructureGroupoid H₁
        inst✝² : HasGroupoid H₂ G₁
        inst✝¹ : ClosedUnderRestriction G₁
        G₂ : StructureGroupoid H₂
        inst✝ : HasGroupoid H₃ G₂
        H : ∀ (e : PartialHomeomorph H₂ H₂), Membership.mem G₂ e → ChartedSpace.LiftPr …
        x✝ : ChartedSpace H₁ H₃ := ChartedSpace.comp H₁ H₂ H₃
        e : PartialHomeomorph H₃ H₂
        he : Membership.mem (atlas H₂ H₃) e
        f : PartialHomeomorph H₂ H₁
        hf : Membership.mem (atlas H₁ H₂) f
        e' : PartialHomeomorph H₃ H₂
        he' : Membership.mem (atlas H₂ H₃) e'
        f' : PartialHomeomorph H₂ H₁
        hf' : Membership.mem (atlas H₁ H₂) f'
        x : H₁
        hx : And (And (Membership.mem f.target x) (Membership.mem e.target (↑f.symm x) …
        hxs : Membership.mem (Set.preimage (↑f.symm) (e.symm.trans e').source) x
        hxs' : Membership.mem (Inter.inter f.target (Set.preimage (↑f.symm) (Inter.int …
        φ : PartialHomeomorph H₁ H₁
        hφG₁ : Membership.mem G₁ φ
        hφ_dom : Membership.mem φ.source x
        hφ : Set.EqOn (↑(f.symm.trans (e.symm.trans (e'.trans f')))) (↑φ.toPartialEqui …
        ⊢ Exists fun s => And (IsOpen s) (And (Membership.mem s x) (Membership.mem G₁  …
      -/
      simp_rw [PartialHomeomorph.trans_symm_eq_symm_trans_symm, PartialHomeomorph.trans_assoc]
      have hs : IsOpen (f.symm ≫ₕ e.symm ≫ₕ e' ≫ₕ f').source :=
        (f.symm ≫ₕ e.symm ≫ₕ e' ≫ₕ f').open_source
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
        H₁ : Type u_6
        inst✝⁷ : TopologicalSpace H₁
        H₂ : Type u_7
        inst✝⁶ : TopologicalSpace H₂
        H₃ : Type u_8
        inst✝⁵ : TopologicalSpace H₃
        inst✝⁴ : ChartedSpace H₁ H₂
        inst✝³ : ChartedSpace H₂ H₃
        G₁ : StructureGroupoid H₁
        inst✝² : HasGroupoid H₂ G₁
        inst✝¹ : ClosedUnderRestriction G₁
        G₂ : StructureGroupoid H₂
        inst✝ : HasGroupoid H₃ G₂
        H : ∀ (e : PartialHomeomorph H₂ H₂), Membership.mem G₂ e → ChartedSpace.LiftPr …
        x✝ : ChartedSpace H₁ H₃ := ChartedSpace.comp H₁ H₂ H₃
        e : PartialHomeomorph H₃ H₂
        he : Membership.mem (atlas H₂ H₃) e
        f : PartialHomeomorph H₂ H₁
        hf : Membership.mem (atlas H₁ H₂) f
        e' : PartialHomeomorph H₃ H₂
        he' : Membership.mem (atlas H₂ H₃) e'
        f' : PartialHomeomorph H₂ H₁
        hf' : Membership.mem (atlas H₁ H₂) f'
        x : H₁
        hx : And (And (Membership.mem f.target x) (Membership.mem e.target (↑f.symm x) …
        hxs : Membership.mem (Set.preimage (↑f.symm) (e.symm.trans e').source) x
        hxs' : Membership.mem (Inter.inter f.target (Set.preimage (↑f.symm) (Inter.int …
        φ : PartialHomeomorph H₁ H₁
        hφG₁ : Membership.mem G₁ φ
        hφ_dom : Membership.mem φ.source x
        hφ : Set.EqOn (↑(f.symm.trans (e.symm.trans (e'.trans f')))) (↑φ.toPartialEqui …
        hs : IsOpen (f.symm.trans (e.symm.trans (e'.trans f'))).source
        ⊢ Exists fun s => And (IsOpen s) (And (Membership.mem s x) (Membership.mem G₁  …
      -/
      refine ⟨_, hs.inter φ.open_source, ?_, ?_⟩
        /-
          case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_1
          H₁ : Type u_6
          inst✝⁷ : TopologicalSpace H₁
          H₂ : Type u_7
          inst✝⁶ : TopologicalSpace H₂
          H₃ : Type u_8
          inst✝⁵ : TopologicalSpace H₃
          inst✝⁴ : ChartedSpace H₁ H₂
          inst✝³ : ChartedSpace H₂ H₃
          G₁ : StructureGroupoid H₁
          inst✝² : HasGroupoid H₂ G₁
          inst✝¹ : ClosedUnderRestriction G₁
          G₂ : StructureGroupoid H₂
          inst✝ : HasGroupoid H₃ G₂
          H : ∀ (e : PartialHomeomorph H₂ H₂), Membership.mem G₂ e → ChartedSpace.LiftPr …
          x✝ : ChartedSpace H₁ H₃ := ChartedSpace.comp H₁ H₂ H₃
          e : PartialHomeomorph H₃ H₂
          he : Membership.mem (atlas H₂ H₃) e
          f : PartialHomeomorph H₂ H₁
          hf : Membership.mem (atlas H₁ H₂) f
          e' : PartialHomeomorph H₃ H₂
          he' : Membership.mem (atlas H₂ H₃) e'
          f' : PartialHomeomorph H₂ H₁
          hf' : Membership.mem (atlas H₁ H₂) f'
          x : H₁
          hx : And (And (Membership.mem f.target x) (Membership.mem e.target (↑f.symm x) …
          hxs : Membership.mem (Set.preimage (↑f.symm) (e.symm.trans e').source) x
          hxs' : Membership.mem (Inter.inter f.target (Set.preimage (↑f.symm) (Inter.int …
          φ : PartialHomeomorph H₁ H₁
          hφG₁ : Membership.mem G₁ φ
          hφ_dom : Membership.mem φ.source x
          hφ : Set.EqOn (↑(f.symm.trans (e.symm.trans (e'.trans f')))) (↑φ.toPartialEqui …
          hs : IsOpen (f.symm.trans (e.symm.trans (e'.trans f'))).source
          ⊢ Membership.mem (Inter.inter (f.symm.trans (e.symm.trans (e'.trans f'))).sour …
        -/
      · simp only [hx, hφ_dom, mfld_simps]
        /-
          🎉 no goals
        -/
        /-
          case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_2
          H₁ : Type u_6
          inst✝⁷ : TopologicalSpace H₁
          H₂ : Type u_7
          inst✝⁶ : TopologicalSpace H₂
          H₃ : Type u_8
          inst✝⁵ : TopologicalSpace H₃
          inst✝⁴ : ChartedSpace H₁ H₂
          inst✝³ : ChartedSpace H₂ H₃
          G₁ : StructureGroupoid H₁
          inst✝² : HasGroupoid H₂ G₁
          inst✝¹ : ClosedUnderRestriction G₁
          G₂ : StructureGroupoid H₂
          inst✝ : HasGroupoid H₃ G₂
          H : ∀ (e : PartialHomeomorph H₂ H₂), Membership.mem G₂ e → ChartedSpace.LiftPr …
          x✝ : ChartedSpace H₁ H₃ := ChartedSpace.comp H₁ H₂ H₃
          e : PartialHomeomorph H₃ H₂
          he : Membership.mem (atlas H₂ H₃) e
          f : PartialHomeomorph H₂ H₁
          hf : Membership.mem (atlas H₁ H₂) f
          e' : PartialHomeomorph H₃ H₂
          he' : Membership.mem (atlas H₂ H₃) e'
          f' : PartialHomeomorph H₂ H₁
          hf' : Membership.mem (atlas H₁ H₂) f'
          x : H₁
          hx : And (And (Membership.mem f.target x) (Membership.mem e.target (↑f.symm x) …
          hxs : Membership.mem (Set.preimage (↑f.symm) (e.symm.trans e').source) x
          hxs' : Membership.mem (Inter.inter f.target (Set.preimage (↑f.symm) (Inter.int …
          φ : PartialHomeomorph H₁ H₁
          hφG₁ : Membership.mem G₁ φ
          hφ_dom : Membership.mem φ.source x
          hφ : Set.EqOn (↑(f.symm.trans (e.symm.trans (e'.trans f')))) (↑φ.toPartialEqui …
          hs : IsOpen (f.symm.trans (e.symm.trans (e'.trans f'))).source
          ⊢ Membership.mem G₁ ((f.symm.trans (e.symm.trans (e'.trans f'))).restr (Inter. …
        -/
      · refine G₁.mem_of_eqOnSource (closedUnderRestriction' hφG₁ hs) ?_
        /-
          case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_2
          H₁ : Type u_6
          inst✝⁷ : TopologicalSpace H₁
          H₂ : Type u_7
          inst✝⁶ : TopologicalSpace H₂
          H₃ : Type u_8
          inst✝⁵ : TopologicalSpace H₃
          inst✝⁴ : ChartedSpace H₁ H₂
          inst✝³ : ChartedSpace H₂ H₃
          G₁ : StructureGroupoid H₁
          inst✝² : HasGroupoid H₂ G₁
          inst✝¹ : ClosedUnderRestriction G₁
          G₂ : StructureGroupoid H₂
          inst✝ : HasGroupoid H₃ G₂
          H : ∀ (e : PartialHomeomorph H₂ H₂), Membership.mem G₂ e → ChartedSpace.LiftPr …
          x✝ : ChartedSpace H₁ H₃ := ChartedSpace.comp H₁ H₂ H₃
          e : PartialHomeomorph H₃ H₂
          he : Membership.mem (atlas H₂ H₃) e
          f : PartialHomeomorph H₂ H₁
          hf : Membership.mem (atlas H₁ H₂) f
          e' : PartialHomeomorph H₃ H₂
          he' : Membership.mem (atlas H₂ H₃) e'
          f' : PartialHomeomorph H₂ H₁
          hf' : Membership.mem (atlas H₁ H₂) f'
          x : H₁
          hx : And (And (Membership.mem f.target x) (Membership.mem e.target (↑f.symm x) …
          hxs : Membership.mem (Set.preimage (↑f.symm) (e.symm.trans e').source) x
          hxs' : Membership.mem (Inter.inter f.target (Set.preimage (↑f.symm) (Inter.int …
          φ : PartialHomeomorph H₁ H₁
          hφG₁ : Membership.mem G₁ φ
          hφ_dom : Membership.mem φ.source x
          hφ : Set.EqOn (↑(f.symm.trans (e.symm.trans (e'.trans f')))) (↑φ.toPartialEqui …
          hs : IsOpen (f.symm.trans (e.symm.trans (e'.trans f'))).source
          ⊢ HasEquiv.Equiv ((f.symm.trans (e.symm.trans (e'.trans f'))).restr (Inter.int …
        -/
        rw [PartialHomeomorph.restr_source_inter]
        /-
          case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_2
          H₁ : Type u_6
          inst✝⁷ : TopologicalSpace H₁
          H₂ : Type u_7
          inst✝⁶ : TopologicalSpace H₂
          H₃ : Type u_8
          inst✝⁵ : TopologicalSpace H₃
          inst✝⁴ : ChartedSpace H₁ H₂
          inst✝³ : ChartedSpace H₂ H₃
          G₁ : StructureGroupoid H₁
          inst✝² : HasGroupoid H₂ G₁
          inst✝¹ : ClosedUnderRestriction G₁
          G₂ : StructureGroupoid H₂
          inst✝ : HasGroupoid H₃ G₂
          H : ∀ (e : PartialHomeomorph H₂ H₂), Membership.mem G₂ e → ChartedSpace.LiftPr …
          x✝ : ChartedSpace H₁ H₃ := ChartedSpace.comp H₁ H₂ H₃
          e : PartialHomeomorph H₃ H₂
          he : Membership.mem (atlas H₂ H₃) e
          f : PartialHomeomorph H₂ H₁
          hf : Membership.mem (atlas H₁ H₂) f
          e' : PartialHomeomorph H₃ H₂
          he' : Membership.mem (atlas H₂ H₃) e'
          f' : PartialHomeomorph H₂ H₁
          hf' : Membership.mem (atlas H₁ H₂) f'
          x : H₁
          hx : And (And (Membership.mem f.target x) (Membership.mem e.target (↑f.symm x) …
          hxs : Membership.mem (Set.preimage (↑f.symm) (e.symm.trans e').source) x
          hxs' : Membership.mem (Inter.inter f.target (Set.preimage (↑f.symm) (Inter.int …
          φ : PartialHomeomorph H₁ H₁
          hφG₁ : Membership.mem G₁ φ
          hφ_dom : Membership.mem φ.source x
          hφ : Set.EqOn (↑(f.symm.trans (e.symm.trans (e'.trans f')))) (↑φ.toPartialEqui …
          hs : IsOpen (f.symm.trans (e.symm.trans (e'.trans f'))).source
          ⊢ HasEquiv.Equiv ((f.symm.trans (e.symm.trans (e'.trans f'))).restr φ.source)  …
        -/
        refine PartialHomeomorph.Set.EqOn.restr_eqOn_source (hφ.mono ?_)
        /-
          case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_2
          H₁ : Type u_6
          inst✝⁷ : TopologicalSpace H₁
          H₂ : Type u_7
          inst✝⁶ : TopologicalSpace H₂
          H₃ : Type u_8
          inst✝⁵ : TopologicalSpace H₃
          inst✝⁴ : ChartedSpace H₁ H₂
          inst✝³ : ChartedSpace H₂ H₃
          G₁ : StructureGroupoid H₁
          inst✝² : HasGroupoid H₂ G₁
          inst✝¹ : ClosedUnderRestriction G₁
          G₂ : StructureGroupoid H₂
          inst✝ : HasGroupoid H₃ G₂
          H : ∀ (e : PartialHomeomorph H₂ H₂), Membership.mem G₂ e → ChartedSpace.LiftPr …
          x✝ : ChartedSpace H₁ H₃ := ChartedSpace.comp H₁ H₂ H₃
          e : PartialHomeomorph H₃ H₂
          he : Membership.mem (atlas H₂ H₃) e
          f : PartialHomeomorph H₂ H₁
          hf : Membership.mem (atlas H₁ H₂) f
          e' : PartialHomeomorph H₃ H₂
          he' : Membership.mem (atlas H₂ H₃) e'
          f' : PartialHomeomorph H₂ H₁
          hf' : Membership.mem (atlas H₁ H₂) f'
          x : H₁
          hx : And (And (Membership.mem f.target x) (Membership.mem e.target (↑f.symm x) …
          hxs : Membership.mem (Set.preimage (↑f.symm) (e.symm.trans e').source) x
          hxs' : Membership.mem (Inter.inter f.target (Set.preimage (↑f.symm) (Inter.int …
          φ : PartialHomeomorph H₁ H₁
          hφG₁ : Membership.mem G₁ φ
          hφ_dom : Membership.mem φ.source x
          hφ : Set.EqOn (↑(f.symm.trans (e.symm.trans (e'.trans f')))) (↑φ.toPartialEqui …
          hs : IsOpen (f.symm.trans (e.symm.trans (e'.trans f'))).source
          ⊢ HasSubset.Subset (Inter.inter (f.symm.trans (e.symm.trans (e'.trans f'))).so …
        -/
        mfld_set_tac }
        /-
          🎉 no goals
        -/


