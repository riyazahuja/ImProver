/-- A preirreducible set `s` is one where there is no non-trivial pair of disjoint opens on `s`. -/
def IsPreirreducible (s : Set X) : Prop :=
  ∀ u v : Set X, IsOpen u → IsOpen v → (s ∩ u).Nonempty → (s ∩ v).Nonempty → (s ∩ (u ∩ v)).Nonempty


/-- An irreducible set `s` is one that is nonempty and
where there is no non-trivial pair of disjoint opens on `s`. -/
def IsIrreducible (s : Set X) : Prop :=
  s.Nonempty ∧ IsPreirreducible s


theorem IsIrreducible.nonempty (h : IsIrreducible s) : s.Nonempty :=
  h.1


theorem IsIrreducible.isPreirreducible (h : IsIrreducible s) : IsPreirreducible s :=
  h.2


theorem isPreirreducible_empty : IsPreirreducible (∅ : Set X) := fun _ _ _ _ _ ⟨_, h1, _⟩ =>
  h1.elim


theorem Set.Subsingleton.isPreirreducible (hs : s.Subsingleton) : IsPreirreducible s :=
  fun _u _v _ _ ⟨_x, hxs, hxu⟩ ⟨y, hys, hyv⟩ => ⟨y, hys, hs hxs hys ▸ hxu, hyv⟩


theorem isPreirreducible_singleton {x} : IsPreirreducible ({x} : Set X) :=
  subsingleton_singleton.isPreirreducible


theorem isIrreducible_singleton {x} : IsIrreducible ({x} : Set X) :=
  ⟨singleton_nonempty x, isPreirreducible_singleton⟩


theorem isPreirreducible_iff_closure : IsPreirreducible (closure s) ↔ IsPreirreducible s :=
  forall₄_congr fun u v hu hv => by
    /-
      X : Type u_1
      inst✝ : TopologicalSpace X
      s u v : Set X
      hu : IsOpen u
      hv : IsOpen v
      ⊢ Iff ((Inter.inter (closure s) u).Nonempty → (Inter.inter (closure s) v).None …
    -/
    iterate 3 rw [closure_inter_open_nonempty_iff]
    /-
      X : Type u_1
      inst✝ : TopologicalSpace X
      s u v : Set X
      hu : IsOpen u
      hv : IsOpen v
      ⊢ IsOpen (Inter.inter u v)
    -/
    exacts [hu.inter hv, hv, hu]
    /-
      🎉 no goals
    -/


theorem isIrreducible_iff_closure : IsIrreducible (closure s) ↔ IsIrreducible s :=
  and_congr closure_nonempty_iff isPreirreducible_iff_closure


protected alias ⟨_, IsPreirreducible.closure⟩ := isPreirreducible_iff_closure


protected alias ⟨_, IsIrreducible.closure⟩ := isIrreducible_iff_closure


theorem exists_preirreducible (s : Set X) (H : IsPreirreducible s) :
    ∃ t : Set X, IsPreirreducible t ∧ s ⊆ t ∧ ∀ u, IsPreirreducible u → t ⊆ u → u = t :=
  let ⟨m, hsm, hm⟩ :=
    zorn_subset_nonempty { t : Set X | IsPreirreducible t }
      (fun c hc hcc _ =>
        ⟨⋃₀ c, fun u v hu hv ⟨y, hy, hyu⟩ ⟨x, hx, hxv⟩ =>
          let ⟨p, hpc, hyp⟩ := mem_sUnion.1 hy
          let ⟨q, hqc, hxq⟩ := mem_sUnion.1 hx
          Or.casesOn (hcc.total hpc hqc)
            (fun hpq : p ⊆ q =>
              let ⟨x, hxp, hxuv⟩ := hc hqc u v hu hv ⟨y, hpq hyp, hyu⟩ ⟨x, hxq, hxv⟩
              ⟨x, mem_sUnion_of_mem hxp hqc, hxuv⟩)
            fun hqp : q ⊆ p =>
            let ⟨x, hxp, hxuv⟩ := hc hpc u v hu hv ⟨y, hyp, hyu⟩ ⟨x, hqp hxq, hxv⟩
            ⟨x, mem_sUnion_of_mem hxp hpc, hxuv⟩,
          fun _ hxc => subset_sUnion_of_mem hxc⟩)
      s H
  ⟨m, hm.prop, hsm, fun _u hu hmu => (hm.eq_of_subset hu hmu).symm⟩


/-- The set of irreducible components of a topological space. -/
def irreducibleComponents (X : Type*) [TopologicalSpace X] : Set (Set X) :=
  {s | Maximal IsIrreducible s}


theorem isClosed_of_mem_irreducibleComponents (s) (H : s ∈ irreducibleComponents X) :
    IsClosed s := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    H : Membership.mem (irreducibleComponents X) s
    ⊢ IsClosed s
  -/
  rw [← closure_eq_iff_isClosed, eq_comm]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    H : Membership.mem (irreducibleComponents X) s
    ⊢ Eq s (closure s)
  -/
  exact subset_closure.antisymm (H.2 H.1.closure subset_closure)
  /-
    🎉 no goals
  -/


theorem irreducibleComponents_eq_maximals_closed (X : Type*) [TopologicalSpace X] :
    irreducibleComponents X = { s | Maximal (fun x ↦ IsClosed x ∧ IsIrreducible x) s} := by
  /-
    X : Type u_3
    inst✝ : TopologicalSpace X
    ⊢ Eq (irreducibleComponents X) (setOf fun s => Maximal (fun x => And (IsClosed …
  -/
  ext s
  /-
    case h
    X : Type u_3
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Iff (Membership.mem (irreducibleComponents X) s) (Membership.mem (setOf fun  …
  -/
  constructor
    /-
      case h.mp
      X : Type u_3
      inst✝ : TopologicalSpace X
      s : Set X
      ⊢ Membership.mem (irreducibleComponents X) s → Membership.mem (setOf fun s =>  …
    -/
  · intro H
    /-
      case h.mp
      X : Type u_3
      inst✝ : TopologicalSpace X
      s : Set X
      H : Membership.mem (irreducibleComponents X) s
      ⊢ Membership.mem (setOf fun s => Maximal (fun x => And (IsClosed x) (IsIrreduc …
    -/
    exact ⟨⟨isClosed_of_mem_irreducibleComponents _ H, H.1⟩, fun x h e => H.2 h.2 e⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      X : Type u_3
      inst✝ : TopologicalSpace X
      s : Set X
      ⊢ Membership.mem (setOf fun s => Maximal (fun x => And (IsClosed x) (IsIrreduc …
    -/
  · intro H
    /-
      case h.mpr
      X : Type u_3
      inst✝ : TopologicalSpace X
      s : Set X
      H : Membership.mem (setOf fun s => Maximal (fun x => And (IsClosed x) (IsIrred …
      ⊢ Membership.mem (irreducibleComponents X) s
    -/
    refine ⟨H.1.2, fun x h e => ?_⟩
    /-
      case h.mpr
      X : Type u_3
      inst✝ : TopologicalSpace X
      s : Set X
      H : Membership.mem (setOf fun s => Maximal (fun x => And (IsClosed x) (IsIrred …
      x : Set X
      h : IsIrreducible x
      e : LE.le s x
      ⊢ LE.le x s
    -/
    have : closure x ≤ s := H.2 ⟨isClosed_closure, h.closure⟩ (e.trans subset_closure)
    /-
      case h.mpr
      X : Type u_3
      inst✝ : TopologicalSpace X
      s : Set X
      H : Membership.mem (setOf fun s => Maximal (fun x => And (IsClosed x) (IsIrred …
      x : Set X
      h : IsIrreducible x
      e : LE.le s x
      this : LE.le (closure x) s
      ⊢ LE.le x s
    -/
    exact le_trans subset_closure this
    /-
      🎉 no goals
    -/


/-- A maximal irreducible set that contains a given point. -/
def irreducibleComponent (x : X) : Set X :=
  Classical.choose (exists_preirreducible {x} isPreirreducible_singleton)


theorem irreducibleComponent_property (x : X) :
    IsPreirreducible (irreducibleComponent x) ∧
      {x} ⊆ irreducibleComponent x ∧
        ∀ u, IsPreirreducible u → irreducibleComponent x ⊆ u → u = irreducibleComponent x :=
  Classical.choose_spec (exists_preirreducible {x} isPreirreducible_singleton)


theorem mem_irreducibleComponent {x : X} : x ∈ irreducibleComponent x :=
  singleton_subset_iff.1 (irreducibleComponent_property x).2.1


theorem isIrreducible_irreducibleComponent {x : X} : IsIrreducible (irreducibleComponent x) :=
  ⟨⟨x, mem_irreducibleComponent⟩, (irreducibleComponent_property x).1⟩


theorem eq_irreducibleComponent {x : X} :
    IsPreirreducible s → irreducibleComponent x ⊆ s → s = irreducibleComponent x :=
  (irreducibleComponent_property x).2.2 _


theorem irreducibleComponent_mem_irreducibleComponents (x : X) :
    irreducibleComponent x ∈ irreducibleComponents X :=
  ⟨isIrreducible_irreducibleComponent, fun _ h₁ h₂ => (eq_irreducibleComponent h₁.2 h₂).le⟩


theorem isClosed_irreducibleComponent {x : X} : IsClosed (irreducibleComponent x) :=
  isClosed_of_mem_irreducibleComponents _ (irreducibleComponent_mem_irreducibleComponents x)


/-- A preirreducible space is one where there is no non-trivial pair of disjoint opens. -/
class PreirreducibleSpace (X : Type*) [TopologicalSpace X] : Prop where
  /-- In a preirreducible space, `Set.univ` is a preirreducible set. -/
  isPreirreducible_univ : IsPreirreducible (univ : Set X)


/-- An irreducible space is one that is nonempty
and where there is no non-trivial pair of disjoint opens. -/
class IrreducibleSpace (X : Type*) [TopologicalSpace X] extends PreirreducibleSpace X : Prop where
  toNonempty : Nonempty X

-- see Note [lower instance priority]

theorem IrreducibleSpace.isIrreducible_univ (X : Type*) [TopologicalSpace X] [IrreducibleSpace X] :
    IsIrreducible (univ : Set X) :=
  ⟨univ_nonempty, PreirreducibleSpace.isPreirreducible_univ⟩


theorem irreducibleSpace_def (X : Type*) [TopologicalSpace X] :
    IrreducibleSpace X ↔ IsIrreducible (⊤ : Set X) :=
  ⟨@IrreducibleSpace.isIrreducible_univ X _, fun h =>
    haveI : PreirreducibleSpace X := ⟨h.2⟩
    ⟨⟨h.1.some⟩⟩⟩


theorem nonempty_preirreducible_inter [PreirreducibleSpace X] :
    IsOpen s → IsOpen t → s.Nonempty → t.Nonempty → (s ∩ t).Nonempty := by
  simpa only [univ_inter, univ_subset_iff] using
    @PreirreducibleSpace.isPreirreducible_univ X _ _ s t


/-- In a (pre)irreducible space, a nonempty open set is dense. -/
protected theorem IsOpen.dense [PreirreducibleSpace X] (ho : IsOpen s) (hne : s.Nonempty) :
    Dense s :=
  dense_iff_inter_open.2 fun _t hto htne => nonempty_preirreducible_inter hto ho htne hne


theorem IsPreirreducible.image (H : IsPreirreducible s) (f : X → Y) (hf : ContinuousOn f s) :
    IsPreirreducible (f '' s) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    H : IsPreirreducible s
    f : X → Y
    hf : ContinuousOn f s
    ⊢ IsPreirreducible (Set.image f s)
  -/
  rintro u v hu hv ⟨_, ⟨⟨x, hx, rfl⟩, hxu⟩⟩ ⟨_, ⟨⟨y, hy, rfl⟩, hyv⟩⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    H : IsPreirreducible s
    f : X → Y
    hf : ContinuousOn f s
    u v : Set Y
    hu : IsOpen u
    hv : IsOpen v
    x : X
    hx : Membership.mem s x
    hxu : Membership.mem u (f x)
    y : X
    hy : Membership.mem s y
    hyv : Membership.mem v (f y)
    ⊢ (Inter.inter (Set.image f s) (Inter.inter u v)).Nonempty
  -/
  rw [← mem_preimage] at hxu hyv
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    H : IsPreirreducible s
    f : X → Y
    hf : ContinuousOn f s
    u v : Set Y
    hu : IsOpen u
    hv : IsOpen v
    x : X
    hx : Membership.mem s x
    hxu : Membership.mem (Set.preimage f u) x
    y : X
    hy : Membership.mem s y
    hyv : Membership.mem (Set.preimage f v) y
    ⊢ (Inter.inter (Set.image f s) (Inter.inter u v)).Nonempty
  -/
  rcases continuousOn_iff'.1 hf u hu with ⟨u', hu', u'_eq⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    H : IsPreirreducible s
    f : X → Y
    hf : ContinuousOn f s
    u v : Set Y
    hu : IsOpen u
    hv : IsOpen v
    x : X
    hx : Membership.mem s x
    hxu : Membership.mem (Set.preimage f u) x
    y : X
    hy : Membership.mem s y
    hyv : Membership.mem (Set.preimage f v) y
    u' : Set X
    hu' : IsOpen u'
    u'_eq : Eq (Inter.inter (Set.preimage f u) s) (Inter.inter u' s)
    ⊢ (Inter.inter (Set.image f s) (Inter.inter u v)).Nonempty
  -/
  rcases continuousOn_iff'.1 hf v hv with ⟨v', hv', v'_eq⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    H : IsPreirreducible s
    f : X → Y
    hf : ContinuousOn f s
    u v : Set Y
    hu : IsOpen u
    hv : IsOpen v
    x : X
    hx : Membership.mem s x
    hxu : Membership.mem (Set.preimage f u) x
    y : X
    hy : Membership.mem s y
    hyv : Membership.mem (Set.preimage f v) y
    u' : Set X
    hu' : IsOpen u'
    u'_eq : Eq (Inter.inter (Set.preimage f u) s) (Inter.inter u' s)
    v' : Set X
    hv' : IsOpen v'
    v'_eq : Eq (Inter.inter (Set.preimage f v) s) (Inter.inter v' s)
    ⊢ (Inter.inter (Set.image f s) (Inter.inter u v)).Nonempty
  -/
  have := H u' v' hu' hv'
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    H : IsPreirreducible s
    f : X → Y
    hf : ContinuousOn f s
    u v : Set Y
    hu : IsOpen u
    hv : IsOpen v
    x : X
    hx : Membership.mem s x
    hxu : Membership.mem (Set.preimage f u) x
    y : X
    hy : Membership.mem s y
    hyv : Membership.mem (Set.preimage f v) y
    u' : Set X
    hu' : IsOpen u'
    u'_eq : Eq (Inter.inter (Set.preimage f u) s) (Inter.inter u' s)
    v' : Set X
    hv' : IsOpen v'
    v'_eq : Eq (Inter.inter (Set.preimage f v) s) (Inter.inter v' s)
    this : (Inter.inter s u').Nonempty → (Inter.inter s v').Nonempty → (Inter.inte …
    ⊢ (Inter.inter (Set.image f s) (Inter.inter u v)).Nonempty
  -/
  rw [inter_comm s u', ← u'_eq] at this
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    H : IsPreirreducible s
    f : X → Y
    hf : ContinuousOn f s
    u v : Set Y
    hu : IsOpen u
    hv : IsOpen v
    x : X
    hx : Membership.mem s x
    hxu : Membership.mem (Set.preimage f u) x
    y : X
    hy : Membership.mem s y
    hyv : Membership.mem (Set.preimage f v) y
    u' : Set X
    hu' : IsOpen u'
    u'_eq : Eq (Inter.inter (Set.preimage f u) s) (Inter.inter u' s)
    v' : Set X
    hv' : IsOpen v'
    v'_eq : Eq (Inter.inter (Set.preimage f v) s) (Inter.inter v' s)
    this : (Inter.inter (Set.preimage f u) s).Nonempty → (Inter.inter s v').Nonemp …
    ⊢ (Inter.inter (Set.image f s) (Inter.inter u v)).Nonempty
  -/
  rw [inter_comm s v', ← v'_eq] at this
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    H : IsPreirreducible s
    f : X → Y
    hf : ContinuousOn f s
    u v : Set Y
    hu : IsOpen u
    hv : IsOpen v
    x : X
    hx : Membership.mem s x
    hxu : Membership.mem (Set.preimage f u) x
    y : X
    hy : Membership.mem s y
    hyv : Membership.mem (Set.preimage f v) y
    u' : Set X
    hu' : IsOpen u'
    u'_eq : Eq (Inter.inter (Set.preimage f u) s) (Inter.inter u' s)
    v' : Set X
    hv' : IsOpen v'
    v'_eq : Eq (Inter.inter (Set.preimage f v) s) (Inter.inter v' s)
    this : (Inter.inter (Set.preimage f u) s).Nonempty → (Inter.inter (Set.preimag …
    ⊢ (Inter.inter (Set.image f s) (Inter.inter u v)).Nonempty
  -/
  rcases this ⟨x, hxu, hx⟩ ⟨y, hyv, hy⟩ with ⟨x, hxs, hxu', hxv'⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    H : IsPreirreducible s
    f : X → Y
    hf : ContinuousOn f s
    u v : Set Y
    hu : IsOpen u
    hv : IsOpen v
    x✝ : X
    hx : Membership.mem s x✝
    hxu : Membership.mem (Set.preimage f u) x✝
    y : X
    hy : Membership.mem s y
    hyv : Membership.mem (Set.preimage f v) y
    u' : Set X
    hu' : IsOpen u'
    u'_eq : Eq (Inter.inter (Set.preimage f u) s) (Inter.inter u' s)
    v' : Set X
    hv' : IsOpen v'
    v'_eq : Eq (Inter.inter (Set.preimage f v) s) (Inter.inter v' s)
    this : (Inter.inter (Set.preimage f u) s).Nonempty → (Inter.inter (Set.preimag …
    x : X
    hxs : Membership.mem s x
    hxu' : Membership.mem u' x
    hxv' : Membership.mem v' x
    ⊢ (Inter.inter (Set.image f s) (Inter.inter u v)).Nonempty
  -/
  refine ⟨f x, mem_image_of_mem f hxs, ?_, ?_⟩
  all_goals
    rw [← mem_preimage]
    apply mem_of_mem_inter_left
    show x ∈ _ ∩ s
    simp [*]


theorem IsIrreducible.image (H : IsIrreducible s) (f : X → Y) (hf : ContinuousOn f s) :
    IsIrreducible (f '' s) :=
  ⟨H.nonempty.image _, H.isPreirreducible.image f hf⟩


theorem Subtype.preirreducibleSpace (h : IsPreirreducible s) : PreirreducibleSpace s where
  isPreirreducible_univ := by
    /-
      X : Type u_1
      inst✝ : TopologicalSpace X
      s : Set X
      h : IsPreirreducible s
      ⊢ IsPreirreducible Set.univ
    -/
    rintro _ _ ⟨u, hu, rfl⟩ ⟨v, hv, rfl⟩ ⟨⟨x, hxs⟩, -, hxu⟩ ⟨⟨y, hys⟩, -, hyv⟩
    /-
      case intro.intro.intro.intro.intro.mk.intro.intro.mk.intro
      X : Type u_1
      inst✝ : TopologicalSpace X
      s : Set X
      h : IsPreirreducible s
      u : Set X
      hu : IsOpen u
      v : Set X
      hv : IsOpen v
      x : X
      hxs : Membership.mem s x
      hxu : Membership.mem (Set.preimage Subtype.val u) ⟨x, hxs⟩
      y : X
      hys : Membership.mem s y
      hyv : Membership.mem (Set.preimage Subtype.val v) ⟨y, hys⟩
      ⊢ (Inter.inter Set.univ (Inter.inter (Set.preimage Subtype.val u) (Set.preimag …
    -/
    rcases h u v hu hv ⟨x, hxs, hxu⟩ ⟨y, hys, hyv⟩ with ⟨x, hxs, ⟨hxu, hxv⟩⟩
    /-
      case intro.intro.intro.intro.intro.mk.intro.intro.mk.intro.intro.intro.intro
      X : Type u_1
      inst✝ : TopologicalSpace X
      s : Set X
      h : IsPreirreducible s
      u : Set X
      hu : IsOpen u
      v : Set X
      hv : IsOpen v
      x✝ : X
      hxs✝ : Membership.mem s x✝
      hxu✝ : Membership.mem (Set.preimage Subtype.val u) ⟨x✝, hxs✝⟩
      y : X
      hys : Membership.mem s y
      hyv : Membership.mem (Set.preimage Subtype.val v) ⟨y, hys⟩
      x : X
      hxs : Membership.mem s x
      hxu : Membership.mem u x
      hxv : Membership.mem v x
      ⊢ (Inter.inter Set.univ (Inter.inter (Set.preimage Subtype.val u) (Set.preimag …
    -/
    exact ⟨⟨x, hxs⟩, ⟨Set.mem_univ _, ⟨hxu, hxv⟩⟩⟩
    /-
      🎉 no goals
    -/


theorem Subtype.irreducibleSpace (h : IsIrreducible s) : IrreducibleSpace s where
  isPreirreducible_univ :=
    (Subtype.preirreducibleSpace h.isPreirreducible).isPreirreducible_univ
  toNonempty := h.nonempty.to_subtype


/-- An infinite type with cofinite topology is an irreducible topological space. -/
instance (priority := 100) {X} [Infinite X] : IrreducibleSpace (CofiniteTopology X) where
  isPreirreducible_univ u v := by
    /-
      X✝ : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X✝
      inst✝¹ : TopologicalSpace Y
      s t : Set X✝
      X : Type u_3
      inst✝ : Infinite X
      u v : Set (CofiniteTopology X)
      ⊢ IsOpen u → IsOpen v → (Inter.inter Set.univ u).Nonempty → (Inter.inter Set.u …
    -/
    haveI : Infinite (CofiniteTopology X) := ‹_›
    /-
      X✝ : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X✝
      inst✝¹ : TopologicalSpace Y
      s t : Set X✝
      X : Type u_3
      inst✝ : Infinite X
      u v : Set (CofiniteTopology X)
      this : Infinite (CofiniteTopology X)
      ⊢ IsOpen u → IsOpen v → (Inter.inter Set.univ u).Nonempty → (Inter.inter Set.u …
    -/
    simp only [CofiniteTopology.isOpen_iff, univ_inter]
    /-
      X✝ : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X✝
      inst✝¹ : TopologicalSpace Y
      s t : Set X✝
      X : Type u_3
      inst✝ : Infinite X
      u v : Set (CofiniteTopology X)
      this : Infinite (CofiniteTopology X)
      ⊢ (u.Nonempty → (HasCompl.compl u).Finite) → (v.Nonempty → (HasCompl.compl v). …
    -/
    intro hu hv hu' hv'
    /-
      X✝ : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X✝
      inst✝¹ : TopologicalSpace Y
      s t : Set X✝
      X : Type u_3
      inst✝ : Infinite X
      u v : Set (CofiniteTopology X)
      this : Infinite (CofiniteTopology X)
      hu : u.Nonempty → (HasCompl.compl u).Finite
      hv : v.Nonempty → (HasCompl.compl v).Finite
      hu' : u.Nonempty
      hv' : v.Nonempty
      ⊢ (Inter.inter u v).Nonempty
    -/
    simpa only [compl_union, compl_compl] using ((hu hu').union (hv hv')).infinite_compl.nonempty
    /-
      🎉 no goals
    -/
  toNonempty := (inferInstance : Nonempty X)


theorem irreducibleComponents_eq_singleton [IrreducibleSpace X] :
    irreducibleComponents X = {univ} :=
  Set.ext fun _ ↦ IsGreatest.maximal_iff (s := IsIrreducible (X := X))
    ⟨IrreducibleSpace.isIrreducible_univ X, fun _ _ ↦ Set.subset_univ _⟩


/-- A set `s` is irreducible if and only if
for every finite collection of open sets all of whose members intersect `s`,
`s` also intersects the intersection of the entire collection
(i.e., there is an element of `s` contained in every member of the collection). -/
theorem isIrreducible_iff_sInter :
    IsIrreducible s ↔
      ∀ (U : Finset (Set X)), (∀ u ∈ U, IsOpen u) → (∀ u ∈ U, (s ∩ u).Nonempty) →
        (s ∩ ⋂₀ ↑U).Nonempty := by
  classical
  refine ⟨fun h U hu hU => ?_, fun h => ⟨?_, ?_⟩⟩
  · induction U using Finset.induction_on with
    | empty => simpa using h.nonempty
    | @insert u U _ IH =>
      rw [Finset.coe_insert, sInter_insert]
      rw [Finset.forall_mem_insert] at hu hU
      exact h.2 _ _ hu.1 (U.finite_toSet.isOpen_sInter hu.2) hU.1 (IH hu.2 hU.2)
  · simpa using h ∅
  · intro u v hu hv hu' hv'
    simpa [*] using h {u, v}


/-- A set is preirreducible if and only if
for every cover by two closed sets, it is contained in one of the two covering sets. -/
theorem isPreirreducible_iff_isClosed_union_isClosed :
    IsPreirreducible s ↔
      ∀ z₁ z₂ : Set X, IsClosed z₁ → IsClosed z₂ → s ⊆ z₁ ∪ z₂ → s ⊆ z₁ ∨ s ⊆ z₂ := by
  refine compl_surjective.forall.trans <| forall_congr' fun z₁ => compl_surjective.forall.trans <|
    forall_congr' fun z₂ => ?_
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s z₁ z₂ : Set X
    ⊢ Iff (IsOpen (HasCompl.compl z₁) → IsOpen (HasCompl.compl z₂) → (Inter.inter  …
  -/
  simp only [isOpen_compl_iff, ← compl_union, inter_compl_nonempty_iff]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s z₁ z₂ : Set X
    ⊢ Iff (IsClosed z₁ → IsClosed z₂ → Not (HasSubset.Subset s z₁) → Not (HasSubse …
  -/
  refine forall₂_congr fun _ _ => ?_
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s z₁ z₂ : Set X
    x✝¹ : IsClosed z₁
    x✝ : IsClosed z₂
    ⊢ Iff (Not (HasSubset.Subset s z₁) → Not (HasSubset.Subset s z₂) → Not (HasSub …
  -/
  rw [← and_imp, ← not_or, not_imp_not]
  /-
    🎉 no goals
  -/

@[deprecated (since := "2024-11-19")] alias
isPreirreducible_iff_closed_union_closed := isPreirreducible_iff_isClosed_union_isClosed


/-- A set is irreducible if and only if for every cover by a finite collection of closed sets, it is
contained in one of the members of the collection. -/
theorem isIrreducible_iff_sUnion_isClosed :
    IsIrreducible s ↔
      ∀ t : Finset (Set X), (∀ z ∈ t, IsClosed z) → (s ⊆ ⋃₀ ↑t) → ∃ z ∈ t, s ⊆ z := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Iff (IsIrreducible s) (∀ (t : Finset (Set X)), (∀ (z : Set X), Membership.me …
  -/
  simp only [isIrreducible_iff_sInter]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Iff (∀ (U : Finset (Set X)), (∀ (u : Set X), Membership.mem U u → IsOpen u)  …
  -/
  refine ((@compl_involutive (Set X) _).toPerm _).finsetCongr.forall_congr fun {t} => ?_
  simp_rw [Equiv.finsetCongr_apply, Finset.forall_mem_map, Finset.mem_map, Finset.coe_map,
    sUnion_image, Equiv.coe_toEmbedding, Function.Involutive.coe_toPerm, isClosed_compl_iff,
    exists_exists_and_eq_and]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    t : Finset (Set X)
    ⊢ Iff ((∀ (u : Set X), Membership.mem t u → IsOpen u) → (∀ (u : Set X), Member …
  -/
  refine forall_congr' fun _ => Iff.trans ?_ not_imp_not
  simp only [not_exists, not_and, ← compl_iInter₂, ← sInter_eq_biInter,
    subset_compl_iff_disjoint_right, not_disjoint_iff_nonempty_inter]


@[deprecated (since := "2024-11-19")] alias
isIrreducible_iff_sUnion_closed := isIrreducible_iff_sUnion_isClosed


/-- A nonempty open subset of a preirreducible subspace is dense in the subspace. -/
theorem subset_closure_inter_of_isPreirreducible_of_isOpen {S U : Set X} (hS : IsPreirreducible S)
    (hU : IsOpen U) (h : (S ∩ U).Nonempty) : S ⊆ closure (S ∩ U) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    S U : Set X
    hS : IsPreirreducible S
    hU : IsOpen U
    h : (Inter.inter S U).Nonempty
    ⊢ HasSubset.Subset S (closure (Inter.inter S U))
  -/
  by_contra h'
  obtain ⟨x, h₁, h₂, h₃⟩ :=
    hS _ (closure (S ∩ U))ᶜ hU isClosed_closure.isOpen_compl h (inter_compl_nonempty_iff.mpr h')
  /-
    case intro.intro.intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    S U : Set X
    hS : IsPreirreducible S
    hU : IsOpen U
    h : (Inter.inter S U).Nonempty
    h' : Not (HasSubset.Subset S (closure (Inter.inter S U)))
    x : X
    h₁ : Membership.mem S x
    h₂ : Membership.mem U x
    h₃ : Membership.mem (HasCompl.compl (closure (Inter.inter S U))) x
    ⊢ False
  -/
  exact h₃ (subset_closure ⟨h₁, h₂⟩)
  /-
    🎉 no goals
  -/


/-- If `∅ ≠ U ⊆ S ⊆ t` such that `U` is open and `t` is preirreducible, then `S` is irreducible. -/
theorem IsPreirreducible.subset_irreducible {S U : Set X} (ht : IsPreirreducible t)
    (hU : U.Nonempty) (hU' : IsOpen U) (h₁ : U ⊆ S) (h₂ : S ⊆ t) : IsIrreducible S := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    t S U : Set X
    ht : IsPreirreducible t
    hU : U.Nonempty
    hU' : IsOpen U
    h₁ : HasSubset.Subset U S
    h₂ : HasSubset.Subset S t
    ⊢ IsIrreducible S
  -/
  obtain ⟨z, hz⟩ := hU
  /-
    case intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    t S U : Set X
    ht : IsPreirreducible t
    hU' : IsOpen U
    h₁ : HasSubset.Subset U S
    h₂ : HasSubset.Subset S t
    z : X
    hz : Membership.mem U z
    ⊢ IsIrreducible S
  -/
  replace ht : IsIrreducible t := ⟨⟨z, h₂ (h₁ hz)⟩, ht⟩
  /-
    case intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    t S U : Set X
    hU' : IsOpen U
    h₁ : HasSubset.Subset U S
    h₂ : HasSubset.Subset S t
    z : X
    hz : Membership.mem U z
    ht : IsIrreducible t
    ⊢ IsIrreducible S
  -/
  refine ⟨⟨z, h₁ hz⟩, ?_⟩
  /-
    case intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    t S U : Set X
    hU' : IsOpen U
    h₁ : HasSubset.Subset U S
    h₂ : HasSubset.Subset S t
    z : X
    hz : Membership.mem U z
    ht : IsIrreducible t
    ⊢ IsPreirreducible S
  -/
  rintro u v hu hv ⟨x, hx, hx'⟩ ⟨y, hy, hy'⟩
  classical
  obtain ⟨x, -, hx'⟩ : Set.Nonempty (t ∩ ⋂₀ ↑({U, u, v} : Finset (Set X))) := by
    refine isIrreducible_iff_sInter.mp ht {U, u, v} ?_ ?_
    · simp [*]
    · intro U H
      simp only [Finset.mem_insert, Finset.mem_singleton] at H
      rcases H with (rfl | rfl | rfl)
      exacts [⟨z, h₂ (h₁ hz), hz⟩, ⟨x, h₂ hx, hx'⟩, ⟨y, h₂ hy, hy'⟩]
  replace hx' : x ∈ U ∧ x ∈ u ∧ x ∈ v := by simpa using hx'
  exact ⟨x, h₁ hx'.1, hx'.2⟩


theorem IsPreirreducible.open_subset {U : Set X} (ht : IsPreirreducible t) (hU : IsOpen U)
    (hU' : U ⊆ t) : IsPreirreducible U :=
  U.eq_empty_or_nonempty.elim (fun h => h.symm ▸ isPreirreducible_empty) fun h =>
    (ht.subset_irreducible h hU (fun _ => id) hU').2


theorem IsPreirreducible.interior (ht : IsPreirreducible t) : IsPreirreducible (interior t) :=
  ht.open_subset isOpen_interior interior_subset


theorem IsPreirreducible.preimage (ht : IsPreirreducible t) {f : Y → X}
    (hf : IsOpenEmbedding f) : IsPreirreducible (f ⁻¹' t) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    t : Set X
    ht : IsPreirreducible t
    f : Y → X
    hf : Topology.IsOpenEmbedding f
    ⊢ IsPreirreducible (Set.preimage f t)
  -/
  rintro U V hU hV ⟨x, hx, hx'⟩ ⟨y, hy, hy'⟩
  obtain ⟨_, h₁, ⟨y, h₂, rfl⟩, ⟨y', h₃, h₄⟩⟩ :=
    ht _ _ (hf.isOpenMap _ hU) (hf.isOpenMap _ hV) ⟨f x, hx, Set.mem_image_of_mem f hx'⟩
      ⟨f y, hy, Set.mem_image_of_mem f hy'⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    t : Set X
    ht : IsPreirreducible t
    f : Y → X
    hf : Topology.IsOpenEmbedding f
    U V : Set Y
    hU : IsOpen U
    hV : IsOpen V
    x : Y
    hx : Membership.mem (Set.preimage f t) x
    hx' : Membership.mem U x
    y✝ : Y
    hy : Membership.mem (Set.preimage f t) y✝
    hy' : Membership.mem V y✝
    y : Y
    h₂ : Membership.mem U y
    h₁ : Membership.mem t (f y)
    y' : Y
    h₃ : Membership.mem V y'
    h₄ : Eq (f y') (f y)
    ⊢ (Inter.inter (Set.preimage f t) (Inter.inter U V)).Nonempty
  -/
  cases hf.injective h₄
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refl
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    t : Set X
    ht : IsPreirreducible t
    f : Y → X
    hf : Topology.IsOpenEmbedding f
    U V : Set Y
    hU : IsOpen U
    hV : IsOpen V
    x : Y
    hx : Membership.mem (Set.preimage f t) x
    hx' : Membership.mem U x
    y✝ : Y
    hy : Membership.mem (Set.preimage f t) y✝
    hy' : Membership.mem V y✝
    y : Y
    h₂ : Membership.mem U y
    h₁ : Membership.mem t (f y)
    h₃ : Membership.mem V y
    h₄ : Eq (f y) (f y)
    ⊢ (Inter.inter (Set.preimage f t) (Inter.inter U V)).Nonempty
  -/
  exact ⟨y, h₁, h₂, h₃⟩
  /-
    🎉 no goals
  -/


