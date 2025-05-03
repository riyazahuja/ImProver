scoped[Manifold] infixr:100 " ≫ₕ " => PartialHomeomorph.trans


scoped[Manifold] infixr:100 " ≫ " => PartialEquiv.trans


/-- A structure groupoid is a set of partial homeomorphisms of a topological space stable under
composition and inverse. They appear in the definition of the smoothness class of a manifold. -/
structure StructureGroupoid (H : Type u) [TopologicalSpace H] where
  /-- Members of the structure groupoid are partial homeomorphisms. -/
  members : Set (PartialHomeomorph H H)
  /-- Structure groupoids are stable under composition. -/
  trans' : ∀ e e' : PartialHomeomorph H H, e ∈ members → e' ∈ members → e ≫ₕ e' ∈ members
  /-- Structure groupoids are stable under inverse. -/
  symm' : ∀ e : PartialHomeomorph H H, e ∈ members → e.symm ∈ members
  /-- The identity morphism lies in the structure groupoid. -/
  id_mem' : PartialHomeomorph.refl H ∈ members
  /-- Let `e` be a partial homeomorphism. If for every `x ∈ e.source`, the restriction of e to some
  open set around `x` lies in the groupoid, then `e` lies in the groupoid. -/
  locality' : ∀ e : PartialHomeomorph H H,
    (∀ x ∈ e.source, ∃ s, IsOpen s ∧ x ∈ s ∧ e.restr s ∈ members) → e ∈ members
  /-- Membership in a structure groupoid respects the equivalence of partial homeomorphisms. -/
  mem_of_eqOnSource' : ∀ e e' : PartialHomeomorph H H, e ∈ members → e' ≈ e → e' ∈ members


instance : Membership (PartialHomeomorph H H) (StructureGroupoid H) :=
  ⟨fun (G : StructureGroupoid H) (e : PartialHomeomorph H H) ↦ e ∈ G.members⟩


instance (H : Type u) [TopologicalSpace H] :
    SetLike (StructureGroupoid H) (PartialHomeomorph H H) where
  coe s := s.members
                             /-
                               H✝ : Type u
                               H' : Type u_1
                               M : Type u_2
                               M' : Type u_3
                               M'' : Type u_4
                               inst✝¹ : TopologicalSpace H✝
                               H : Type u
                               inst✝ : TopologicalSpace H
                               N O : StructureGroupoid H
                               h : Eq ((fun s => s.members) N) ((fun s => s.members) O)
                               ⊢ Eq N O
                             -/
  coe_injective' N O h := by cases N; cases O; congr
                                               /-
                                                 🎉 no goals
                                               -/


instance : Min (StructureGroupoid H) :=
  ⟨fun G G' => StructureGroupoid.mk
    (members := G.members ∩ G'.members)
    (trans' := fun e e' he he' =>
      ⟨G.trans' e e' he.left he'.left, G'.trans' e e' he.right he'.right⟩)
    (symm' := fun e he => ⟨G.symm' e he.left, G'.symm' e he.right⟩)
    (id_mem' := ⟨G.id_mem', G'.id_mem'⟩)
    (locality' := by
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        G G' : StructureGroupoid H
        ⊢ ∀ (e : PartialHomeomorph H H), (∀ (x : H), Membership.mem e.source x → Exist …
      -/
      intro e hx
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        G G' : StructureGroupoid H
        e : PartialHomeomorph H H
        hx : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
        ⊢ Membership.mem (Inter.inter G.members G'.members) e
      -/
      apply (mem_inter_iff e G.members G'.members).mpr
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        G G' : StructureGroupoid H
        e : PartialHomeomorph H H
        hx : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
        ⊢ And (Membership.mem G.members e) (Membership.mem G'.members e)
      -/
      refine And.intro (G.locality' e ?_) (G'.locality' e ?_)
      all_goals
        intro x hex
        rcases hx x hex with ⟨s, hs⟩
        use s
        refine And.intro hs.left (And.intro hs.right.left ?_)
        /-
          case h
          H : Type u
          H' : Type u_1
          M : Type u_2
          M' : Type u_3
          M'' : Type u_4
          inst✝ : TopologicalSpace H
          G G' : StructureGroupoid H
          e : PartialHomeomorph H H
          hx : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
          x : H
          hex : Membership.mem e.source x
          s : Set H
          hs : And (IsOpen s) (And (Membership.mem s x) (Membership.mem (Inter.inter G.m …
          ⊢ Membership.mem G.members (e.restr s)
        -/
      · exact hs.right.right.left
        /-
          🎉 no goals
        -/
        /-
          case h
          H : Type u
          H' : Type u_1
          M : Type u_2
          M' : Type u_3
          M'' : Type u_4
          inst✝ : TopologicalSpace H
          G G' : StructureGroupoid H
          e : PartialHomeomorph H H
          hx : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
          x : H
          hex : Membership.mem e.source x
          s : Set H
          hs : And (IsOpen s) (And (Membership.mem s x) (Membership.mem (Inter.inter G.m …
          ⊢ Membership.mem G'.members (e.restr s)
        -/
      · exact hs.right.right.right)
        /-
          🎉 no goals
        -/
    (mem_of_eqOnSource' := fun e e' he hee' =>
      ⟨G.mem_of_eqOnSource' e e' he.left hee', G'.mem_of_eqOnSource' e e' he.right hee'⟩)⟩


instance : InfSet (StructureGroupoid H) :=
  ⟨fun S => StructureGroupoid.mk
    (members := ⋂ s ∈ S, s.members)
    (trans' := by
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        S : Set (StructureGroupoid H)
        ⊢ ∀ (e e' : PartialHomeomorph H H), Membership.mem (Set.iInter fun s => Set.iI …
      -/
      simp only [mem_iInter]
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        S : Set (StructureGroupoid H)
        ⊢ ∀ (e e' : PartialHomeomorph H H), (∀ (i : StructureGroupoid H), Membership.m …
      -/
      intro e e' he he' i hi
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        S : Set (StructureGroupoid H)
        e e' : PartialHomeomorph H H
        he : ∀ (i : StructureGroupoid H), Membership.mem S i → Membership.mem i.member …
        he' : ∀ (i : StructureGroupoid H), Membership.mem S i → Membership.mem i.membe …
        i : StructureGroupoid H
        hi : Membership.mem S i
        ⊢ Membership.mem i.members (e.trans e')
      -/
      exact i.trans' e e' (he i hi) (he' i hi))
      /-
        🎉 no goals
      -/
    (symm' := by
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        S : Set (StructureGroupoid H)
        ⊢ ∀ (e : PartialHomeomorph H H), Membership.mem (Set.iInter fun s => Set.iInte …
      -/
      simp only [mem_iInter]
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        S : Set (StructureGroupoid H)
        ⊢ ∀ (e : PartialHomeomorph H H), (∀ (i : StructureGroupoid H), Membership.mem  …
      -/
      intro e he i hi
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        S : Set (StructureGroupoid H)
        e : PartialHomeomorph H H
        he : ∀ (i : StructureGroupoid H), Membership.mem S i → Membership.mem i.member …
        i : StructureGroupoid H
        hi : Membership.mem S i
        ⊢ Membership.mem i.members e.symm
      -/
      exact i.symm' e (he i hi))
      /-
        🎉 no goals
      -/
    (id_mem' := by
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        S : Set (StructureGroupoid H)
        ⊢ Membership.mem (Set.iInter fun s => Set.iInter fun h => s.members) (PartialH …
      -/
      simp only [mem_iInter]
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        S : Set (StructureGroupoid H)
        ⊢ ∀ (i : StructureGroupoid H), Membership.mem S i → Membership.mem i.members ( …
      -/
      intro i _
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        S : Set (StructureGroupoid H)
        i : StructureGroupoid H
        i✝ : Membership.mem S i
        ⊢ Membership.mem i.members (PartialHomeomorph.refl H)
      -/
      exact i.id_mem')
      /-
        🎉 no goals
      -/
    (locality' := by
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        S : Set (StructureGroupoid H)
        ⊢ ∀ (e : PartialHomeomorph H H), (∀ (x : H), Membership.mem e.source x → Exist …
      -/
      simp only [mem_iInter]
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        S : Set (StructureGroupoid H)
        ⊢ ∀ (e : PartialHomeomorph H H), (∀ (x : H), Membership.mem e.source x → Exist …
      -/
      intro e he i hi
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        S : Set (StructureGroupoid H)
        e : PartialHomeomorph H H
        he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
        i : StructureGroupoid H
        hi : Membership.mem S i
        ⊢ Membership.mem i.members e
      -/
      refine i.locality' e ?_
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        S : Set (StructureGroupoid H)
        e : PartialHomeomorph H H
        he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
        i : StructureGroupoid H
        hi : Membership.mem S i
        ⊢ ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (And ( …
      -/
      intro x hex
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        S : Set (StructureGroupoid H)
        e : PartialHomeomorph H H
        he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
        i : StructureGroupoid H
        hi : Membership.mem S i
        x : H
        hex : Membership.mem e.source x
        ⊢ Exists fun s => And (IsOpen s) (And (Membership.mem s x) (Membership.mem i.m …
      -/
      rcases he x hex with ⟨s, hs⟩
      /-
        case intro
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        S : Set (StructureGroupoid H)
        e : PartialHomeomorph H H
        he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
        i : StructureGroupoid H
        hi : Membership.mem S i
        x : H
        hex : Membership.mem e.source x
        s : Set H
        hs : And (IsOpen s) (And (Membership.mem s x) (∀ (i : StructureGroupoid H), Me …
        ⊢ Exists fun s => And (IsOpen s) (And (Membership.mem s x) (Membership.mem i.m …
      -/
      exact ⟨s, ⟨hs.left, ⟨hs.right.left, hs.right.right i hi⟩⟩⟩)
      /-
        🎉 no goals
      -/
    (mem_of_eqOnSource' := by
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        S : Set (StructureGroupoid H)
        ⊢ ∀ (e e' : PartialHomeomorph H H), Membership.mem (Set.iInter fun s => Set.iI …
      -/
      simp only [mem_iInter]
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        S : Set (StructureGroupoid H)
        ⊢ ∀ (e e' : PartialHomeomorph H H), (∀ (i : StructureGroupoid H), Membership.m …
      -/
      intro e e' he he'e
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        S : Set (StructureGroupoid H)
        e e' : PartialHomeomorph H H
        he : ∀ (i : StructureGroupoid H), Membership.mem S i → Membership.mem i.member …
        he'e : HasEquiv.Equiv e' e
        ⊢ ∀ (i : StructureGroupoid H), Membership.mem S i → Membership.mem i.members e'
      -/
      exact fun i hi => i.mem_of_eqOnSource' e e' (he i hi) he'e)⟩
      /-
        🎉 no goals
      -/


theorem StructureGroupoid.trans (G : StructureGroupoid H) {e e' : PartialHomeomorph H H}
    (he : e ∈ G) (he' : e' ∈ G) : e ≫ₕ e' ∈ G :=
  G.trans' e e' he he'


theorem StructureGroupoid.symm (G : StructureGroupoid H) {e : PartialHomeomorph H H} (he : e ∈ G) :
    e.symm ∈ G :=
  G.symm' e he


theorem StructureGroupoid.id_mem (G : StructureGroupoid H) : PartialHomeomorph.refl H ∈ G :=
  G.id_mem'


theorem StructureGroupoid.locality (G : StructureGroupoid H) {e : PartialHomeomorph H H}
    (h : ∀ x ∈ e.source, ∃ s, IsOpen s ∧ x ∈ s ∧ e.restr s ∈ G) : e ∈ G :=
  G.locality' e h


theorem StructureGroupoid.mem_of_eqOnSource (G : StructureGroupoid H) {e e' : PartialHomeomorph H H}
    (he : e ∈ G) (h : e' ≈ e) : e' ∈ G :=
  G.mem_of_eqOnSource' e e' he h


theorem StructureGroupoid.mem_iff_of_eqOnSource {G : StructureGroupoid H}
    {e e' : PartialHomeomorph H H} (h : e ≈ e') : e ∈ G ↔ e' ∈ G :=
  ⟨fun he ↦ G.mem_of_eqOnSource he (Setoid.symm h), fun he' ↦ G.mem_of_eqOnSource he' h⟩


/-- Partial order on the set of groupoids, given by inclusion of the members of the groupoid. -/
instance StructureGroupoid.partialOrder : PartialOrder (StructureGroupoid H) :=
  PartialOrder.lift StructureGroupoid.members fun a b h ↦ by
    /-
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      a b : StructureGroupoid H
      h : Eq a.members b.members
      ⊢ Eq a b
    -/
    cases a
    /-
      case mk
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      b : StructureGroupoid H
      members✝ : Set (PartialHomeomorph H H)
      trans'✝ : ∀ (e e' : PartialHomeomorph H H), Membership.mem members✝ e → Member …
      symm'✝ : ∀ (e : PartialHomeomorph H H), Membership.mem members✝ e → Membership …
      id_mem'✝ : Membership.mem members✝ (PartialHomeomorph.refl H)
      locality'✝ : ∀ (e : PartialHomeomorph H H), (∀ (x : H), Membership.mem e.sourc …
      mem_of_eqOnSource'✝ : ∀ (e e' : PartialHomeomorph H H), Membership.mem members …
      h : Eq { members := members✝, trans' := trans'✝, symm' := symm'✝, id_mem' := i …
      ⊢ Eq { members := members✝, trans' := trans'✝, symm' := symm'✝, id_mem' := id_ …
    -/
    cases b
    /-
      case mk.mk
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      members✝¹ : Set (PartialHomeomorph H H)
      trans'✝¹ : ∀ (e e' : PartialHomeomorph H H), Membership.mem members✝¹ e → Memb …
      symm'✝¹ : ∀ (e : PartialHomeomorph H H), Membership.mem members✝¹ e → Membersh …
      id_mem'✝¹ : Membership.mem members✝¹ (PartialHomeomorph.refl H)
      locality'✝¹ : ∀ (e : PartialHomeomorph H H), (∀ (x : H), Membership.mem e.sour …
      mem_of_eqOnSource'✝¹ : ∀ (e e' : PartialHomeomorph H H), Membership.mem member …
      members✝ : Set (PartialHomeomorph H H)
      trans'✝ : ∀ (e e' : PartialHomeomorph H H), Membership.mem members✝ e → Member …
      symm'✝ : ∀ (e : PartialHomeomorph H H), Membership.mem members✝ e → Membership …
      id_mem'✝ : Membership.mem members✝ (PartialHomeomorph.refl H)
      locality'✝ : ∀ (e : PartialHomeomorph H H), (∀ (x : H), Membership.mem e.sourc …
      mem_of_eqOnSource'✝ : ∀ (e e' : PartialHomeomorph H H), Membership.mem members …
      h : Eq { members := members✝¹, trans' := trans'✝¹, symm' := symm'✝¹, id_mem' : …
      ⊢ Eq { members := members✝¹, trans' := trans'✝¹, symm' := symm'✝¹, id_mem' :=  …
    -/
    dsimp at h
    /-
      case mk.mk
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      members✝¹ : Set (PartialHomeomorph H H)
      trans'✝¹ : ∀ (e e' : PartialHomeomorph H H), Membership.mem members✝¹ e → Memb …
      symm'✝¹ : ∀ (e : PartialHomeomorph H H), Membership.mem members✝¹ e → Membersh …
      id_mem'✝¹ : Membership.mem members✝¹ (PartialHomeomorph.refl H)
      locality'✝¹ : ∀ (e : PartialHomeomorph H H), (∀ (x : H), Membership.mem e.sour …
      mem_of_eqOnSource'✝¹ : ∀ (e e' : PartialHomeomorph H H), Membership.mem member …
      members✝ : Set (PartialHomeomorph H H)
      trans'✝ : ∀ (e e' : PartialHomeomorph H H), Membership.mem members✝ e → Member …
      symm'✝ : ∀ (e : PartialHomeomorph H H), Membership.mem members✝ e → Membership …
      id_mem'✝ : Membership.mem members✝ (PartialHomeomorph.refl H)
      locality'✝ : ∀ (e : PartialHomeomorph H H), (∀ (x : H), Membership.mem e.sourc …
      mem_of_eqOnSource'✝ : ∀ (e e' : PartialHomeomorph H H), Membership.mem members …
      h : Eq members✝¹ members✝
      ⊢ Eq { members := members✝¹, trans' := trans'✝¹, symm' := symm'✝¹, id_mem' :=  …
    -/
    induction h
    /-
      case mk.mk.refl
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      members✝¹ : Set (PartialHomeomorph H H)
      trans'✝¹ : ∀ (e e' : PartialHomeomorph H H), Membership.mem members✝¹ e → Memb …
      symm'✝¹ : ∀ (e : PartialHomeomorph H H), Membership.mem members✝¹ e → Membersh …
      id_mem'✝¹ : Membership.mem members✝¹ (PartialHomeomorph.refl H)
      locality'✝¹ : ∀ (e : PartialHomeomorph H H), (∀ (x : H), Membership.mem e.sour …
      mem_of_eqOnSource'✝¹ : ∀ (e e' : PartialHomeomorph H H), Membership.mem member …
      members✝ : Set (PartialHomeomorph H H)
      trans'✝ : ∀ (e e' : PartialHomeomorph H H), Membership.mem members✝¹ e → Membe …
      symm'✝ : ∀ (e : PartialHomeomorph H H), Membership.mem members✝¹ e → Membershi …
      id_mem'✝ : Membership.mem members✝¹ (PartialHomeomorph.refl H)
      locality'✝ : ∀ (e : PartialHomeomorph H H), (∀ (x : H), Membership.mem e.sourc …
      mem_of_eqOnSource'✝ : ∀ (e e' : PartialHomeomorph H H), Membership.mem members …
      ⊢ Eq { members := members✝¹, trans' := trans'✝¹, symm' := symm'✝¹, id_mem' :=  …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem StructureGroupoid.le_iff {G₁ G₂ : StructureGroupoid H} : G₁ ≤ G₂ ↔ ∀ e, e ∈ G₁ → e ∈ G₂ :=
  Iff.rfl


/-- The trivial groupoid, containing only the identity (and maps with empty source, as this is
necessary from the definition). -/
def idGroupoid (H : Type u) [TopologicalSpace H] : StructureGroupoid H where
  members := {PartialHomeomorph.refl H} ∪ { e : PartialHomeomorph H H | e.source = ∅ }
  trans' e e' he he' := by
    /-
      H✝ : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝¹ : TopologicalSpace H✝
      H : Type u
      inst✝ : TopologicalSpace H
      e e' : PartialHomeomorph H H
      he : Membership.mem (Union.union (Singleton.singleton (PartialHomeomorph.refl  …
      he' : Membership.mem (Union.union (Singleton.singleton (PartialHomeomorph.refl …
      ⊢ Membership.mem (Union.union (Singleton.singleton (PartialHomeomorph.refl H)) …
    -/
    cases' he with he he
      /-
        case inl
        H✝ : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝¹ : TopologicalSpace H✝
        H : Type u
        inst✝ : TopologicalSpace H
        e e' : PartialHomeomorph H H
        he' : Membership.mem (Union.union (Singleton.singleton (PartialHomeomorph.refl …
        he : Membership.mem (Singleton.singleton (PartialHomeomorph.refl H)) e
        ⊢ Membership.mem (Union.union (Singleton.singleton (PartialHomeomorph.refl H)) …
      -/
    · simpa only [mem_singleton_iff.1 he, refl_trans]
      /-
        🎉 no goals
      -/
      /-
        case inr
        H✝ : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝¹ : TopologicalSpace H✝
        H : Type u
        inst✝ : TopologicalSpace H
        e e' : PartialHomeomorph H H
        he' : Membership.mem (Union.union (Singleton.singleton (PartialHomeomorph.refl …
        he : Membership.mem (setOf fun e => Eq e.source EmptyCollection.emptyCollectio …
        ⊢ Membership.mem (Union.union (Singleton.singleton (PartialHomeomorph.refl H)) …
      -/
    · have : (e ≫ₕ e').source ⊆ e.source := sep_subset _ _
      /-
        case inr
        H✝ : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝¹ : TopologicalSpace H✝
        H : Type u
        inst✝ : TopologicalSpace H
        e e' : PartialHomeomorph H H
        he' : Membership.mem (Union.union (Singleton.singleton (PartialHomeomorph.refl …
        he : Membership.mem (setOf fun e => Eq e.source EmptyCollection.emptyCollectio …
        this : HasSubset.Subset (e.trans e').source e.source
        ⊢ Membership.mem (Union.union (Singleton.singleton (PartialHomeomorph.refl H)) …
      -/
      rw [he] at this
      /-
        case inr
        H✝ : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝¹ : TopologicalSpace H✝
        H : Type u
        inst✝ : TopologicalSpace H
        e e' : PartialHomeomorph H H
        he' : Membership.mem (Union.union (Singleton.singleton (PartialHomeomorph.refl …
        he : Membership.mem (setOf fun e => Eq e.source EmptyCollection.emptyCollectio …
        this : HasSubset.Subset (e.trans e').source EmptyCollection.emptyCollection
        ⊢ Membership.mem (Union.union (Singleton.singleton (PartialHomeomorph.refl H)) …
      -/
      have : e ≫ₕ e' ∈ { e : PartialHomeomorph H H | e.source = ∅ } := eq_bot_iff.2 this
      /-
        case inr
        H✝ : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝¹ : TopologicalSpace H✝
        H : Type u
        inst✝ : TopologicalSpace H
        e e' : PartialHomeomorph H H
        he' : Membership.mem (Union.union (Singleton.singleton (PartialHomeomorph.refl …
        he : Membership.mem (setOf fun e => Eq e.source EmptyCollection.emptyCollectio …
        this✝ : HasSubset.Subset (e.trans e').source EmptyCollection.emptyCollection
        this : Membership.mem (setOf fun e => Eq e.source EmptyCollection.emptyCollect …
        ⊢ Membership.mem (Union.union (Singleton.singleton (PartialHomeomorph.refl H)) …
      -/
      exact (mem_union _ _ _).2 (Or.inr this)
      /-
        🎉 no goals
      -/
  symm' e he := by
    /-
      H✝ : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝¹ : TopologicalSpace H✝
      H : Type u
      inst✝ : TopologicalSpace H
      e : PartialHomeomorph H H
      he : Membership.mem (Union.union (Singleton.singleton (PartialHomeomorph.refl  …
      ⊢ Membership.mem (Union.union (Singleton.singleton (PartialHomeomorph.refl H)) …
    -/
    cases' (mem_union _ _ _).1 he with E E
      /-
        case inl
        H✝ : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝¹ : TopologicalSpace H✝
        H : Type u
        inst✝ : TopologicalSpace H
        e : PartialHomeomorph H H
        he : Membership.mem (Union.union (Singleton.singleton (PartialHomeomorph.refl  …
        E : Membership.mem (Singleton.singleton (PartialHomeomorph.refl H)) e
        ⊢ Membership.mem (Union.union (Singleton.singleton (PartialHomeomorph.refl H)) …
      -/
    · simp [mem_singleton_iff.mp E]
      /-
        🎉 no goals
      -/
      /-
        case inr
        H✝ : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝¹ : TopologicalSpace H✝
        H : Type u
        inst✝ : TopologicalSpace H
        e : PartialHomeomorph H H
        he : Membership.mem (Union.union (Singleton.singleton (PartialHomeomorph.refl  …
        E : Membership.mem (setOf fun e => Eq e.source EmptyCollection.emptyCollection …
        ⊢ Membership.mem (Union.union (Singleton.singleton (PartialHomeomorph.refl H)) …
      -/
    · right
      /-
        case inr.h
        H✝ : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝¹ : TopologicalSpace H✝
        H : Type u
        inst✝ : TopologicalSpace H
        e : PartialHomeomorph H H
        he : Membership.mem (Union.union (Singleton.singleton (PartialHomeomorph.refl  …
        E : Membership.mem (setOf fun e => Eq e.source EmptyCollection.emptyCollection …
        ⊢ Membership.mem (setOf fun e => Eq e.source EmptyCollection.emptyCollection)  …
      -/
      simpa only [e.toPartialEquiv.image_source_eq_target.symm, mfld_simps] using E
      /-
        🎉 no goals
      -/
  id_mem' := mem_union_left _ rfl
  locality' e he := by
    /-
      H✝ : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝¹ : TopologicalSpace H✝
      H : Type u
      inst✝ : TopologicalSpace H
      e : PartialHomeomorph H H
      he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
      ⊢ Membership.mem (Union.union (Singleton.singleton (PartialHomeomorph.refl H)) …
    -/
    rcases e.source.eq_empty_or_nonempty with h | h
      /-
        case inl
        H✝ : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝¹ : TopologicalSpace H✝
        H : Type u
        inst✝ : TopologicalSpace H
        e : PartialHomeomorph H H
        he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
        h : Eq e.source EmptyCollection.emptyCollection
        ⊢ Membership.mem (Union.union (Singleton.singleton (PartialHomeomorph.refl H)) …
      -/
    · right
      /-
        case inl.h
        H✝ : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝¹ : TopologicalSpace H✝
        H : Type u
        inst✝ : TopologicalSpace H
        e : PartialHomeomorph H H
        he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
        h : Eq e.source EmptyCollection.emptyCollection
        ⊢ Membership.mem (setOf fun e => Eq e.source EmptyCollection.emptyCollection) e
      -/
      exact h
      /-
        🎉 no goals
      -/
      /-
        case inr
        H✝ : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝¹ : TopologicalSpace H✝
        H : Type u
        inst✝ : TopologicalSpace H
        e : PartialHomeomorph H H
        he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
        h : e.source.Nonempty
        ⊢ Membership.mem (Union.union (Singleton.singleton (PartialHomeomorph.refl H)) …
      -/
    · left
      /-
        case inr.h
        H✝ : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝¹ : TopologicalSpace H✝
        H : Type u
        inst✝ : TopologicalSpace H
        e : PartialHomeomorph H H
        he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
        h : e.source.Nonempty
        ⊢ Membership.mem (Singleton.singleton (PartialHomeomorph.refl H)) e
      -/
      rcases h with ⟨x, hx⟩
      /-
        case inr.h.intro
        H✝ : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝¹ : TopologicalSpace H✝
        H : Type u
        inst✝ : TopologicalSpace H
        e : PartialHomeomorph H H
        he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
        x : H
        hx : Membership.mem e.source x
        ⊢ Membership.mem (Singleton.singleton (PartialHomeomorph.refl H)) e
      -/
      rcases he x hx with ⟨s, open_s, xs, hs⟩
      have x's : x ∈ (e.restr s).source := by
        rw [restr_source, open_s.interior_eq]
        exact ⟨hx, xs⟩
      /-
        case inr.h.intro.intro.intro.intro
        H✝ : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝¹ : TopologicalSpace H✝
        H : Type u
        inst✝ : TopologicalSpace H
        e : PartialHomeomorph H H
        he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
        x : H
        hx : Membership.mem e.source x
        s : Set H
        open_s : IsOpen s
        xs : Membership.mem s x
        hs : Membership.mem (Union.union (Singleton.singleton (PartialHomeomorph.refl  …
        x's : Membership.mem (e.restr s).source x
        ⊢ Membership.mem (Singleton.singleton (PartialHomeomorph.refl H)) e
      -/
      cases' hs with hs hs
      · replace hs : PartialHomeomorph.restr e s = PartialHomeomorph.refl H := by
          simpa only using hs
        have : (e.restr s).source = univ := by
          rw [hs]
          simp
        /-
          case inr.h.intro.intro.intro.intro.inl
          H✝ : Type u
          H' : Type u_1
          M : Type u_2
          M' : Type u_3
          M'' : Type u_4
          inst✝¹ : TopologicalSpace H✝
          H : Type u
          inst✝ : TopologicalSpace H
          e : PartialHomeomorph H H
          he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
          x : H
          hx : Membership.mem e.source x
          s : Set H
          open_s : IsOpen s
          xs : Membership.mem s x
          x's : Membership.mem (e.restr s).source x
          hs : Eq (e.restr s) (PartialHomeomorph.refl H)
          this : Eq (e.restr s).source Set.univ
          ⊢ Membership.mem (Singleton.singleton (PartialHomeomorph.refl H)) e
        -/
        have : e.toPartialEquiv.source ∩ interior s = univ := this
        have : univ ⊆ interior s := by
          rw [← this]
          exact inter_subset_right
        /-
          case inr.h.intro.intro.intro.intro.inl
          H✝ : Type u
          H' : Type u_1
          M : Type u_2
          M' : Type u_3
          M'' : Type u_4
          inst✝¹ : TopologicalSpace H✝
          H : Type u
          inst✝ : TopologicalSpace H
          e : PartialHomeomorph H H
          he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
          x : H
          hx : Membership.mem e.source x
          s : Set H
          open_s : IsOpen s
          xs : Membership.mem s x
          x's : Membership.mem (e.restr s).source x
          hs : Eq (e.restr s) (PartialHomeomorph.refl H)
          this✝¹ : Eq (e.restr s).source Set.univ
          this✝ : Eq (Inter.inter e.source (interior s)) Set.univ
          this : HasSubset.Subset Set.univ (interior s)
          ⊢ Membership.mem (Singleton.singleton (PartialHomeomorph.refl H)) e
        -/
        have : s = univ := by rwa [open_s.interior_eq, univ_subset_iff] at this
        /-
          case inr.h.intro.intro.intro.intro.inl
          H✝ : Type u
          H' : Type u_1
          M : Type u_2
          M' : Type u_3
          M'' : Type u_4
          inst✝¹ : TopologicalSpace H✝
          H : Type u
          inst✝ : TopologicalSpace H
          e : PartialHomeomorph H H
          he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
          x : H
          hx : Membership.mem e.source x
          s : Set H
          open_s : IsOpen s
          xs : Membership.mem s x
          x's : Membership.mem (e.restr s).source x
          hs : Eq (e.restr s) (PartialHomeomorph.refl H)
          this✝² : Eq (e.restr s).source Set.univ
          this✝¹ : Eq (Inter.inter e.source (interior s)) Set.univ
          this✝ : HasSubset.Subset Set.univ (interior s)
          this : Eq s Set.univ
          ⊢ Membership.mem (Singleton.singleton (PartialHomeomorph.refl H)) e
        -/
        simpa only [this, restr_univ] using hs
        /-
          🎉 no goals
        -/
        /-
          case inr.h.intro.intro.intro.intro.inr
          H✝ : Type u
          H' : Type u_1
          M : Type u_2
          M' : Type u_3
          M'' : Type u_4
          inst✝¹ : TopologicalSpace H✝
          H : Type u
          inst✝ : TopologicalSpace H
          e : PartialHomeomorph H H
          he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
          x : H
          hx : Membership.mem e.source x
          s : Set H
          open_s : IsOpen s
          xs : Membership.mem s x
          x's : Membership.mem (e.restr s).source x
          hs : Membership.mem (setOf fun e => Eq e.source EmptyCollection.emptyCollectio …
          ⊢ Membership.mem (Singleton.singleton (PartialHomeomorph.refl H)) e
        -/
      · exfalso
        /-
          case inr.h.intro.intro.intro.intro.inr
          H✝ : Type u
          H' : Type u_1
          M : Type u_2
          M' : Type u_3
          M'' : Type u_4
          inst✝¹ : TopologicalSpace H✝
          H : Type u
          inst✝ : TopologicalSpace H
          e : PartialHomeomorph H H
          he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
          x : H
          hx : Membership.mem e.source x
          s : Set H
          open_s : IsOpen s
          xs : Membership.mem s x
          x's : Membership.mem (e.restr s).source x
          hs : Membership.mem (setOf fun e => Eq e.source EmptyCollection.emptyCollectio …
          ⊢ False
        -/
        rw [mem_setOf_eq] at hs
        /-
          case inr.h.intro.intro.intro.intro.inr
          H✝ : Type u
          H' : Type u_1
          M : Type u_2
          M' : Type u_3
          M'' : Type u_4
          inst✝¹ : TopologicalSpace H✝
          H : Type u
          inst✝ : TopologicalSpace H
          e : PartialHomeomorph H H
          he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
          x : H
          hx : Membership.mem e.source x
          s : Set H
          open_s : IsOpen s
          xs : Membership.mem s x
          x's : Membership.mem (e.restr s).source x
          hs : Eq (e.restr s).source EmptyCollection.emptyCollection
          ⊢ False
        -/
        rwa [hs] at x's
        /-
          🎉 no goals
        -/
  mem_of_eqOnSource' e e' he he'e := by
    /-
      H✝ : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝¹ : TopologicalSpace H✝
      H : Type u
      inst✝ : TopologicalSpace H
      e e' : PartialHomeomorph H H
      he : Membership.mem (Union.union (Singleton.singleton (PartialHomeomorph.refl  …
      he'e : HasEquiv.Equiv e' e
      ⊢ Membership.mem (Union.union (Singleton.singleton (PartialHomeomorph.refl H)) …
    -/
    cases' he with he he
      /-
        case inl
        H✝ : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝¹ : TopologicalSpace H✝
        H : Type u
        inst✝ : TopologicalSpace H
        e e' : PartialHomeomorph H H
        he'e : HasEquiv.Equiv e' e
        he : Membership.mem (Singleton.singleton (PartialHomeomorph.refl H)) e
        ⊢ Membership.mem (Union.union (Singleton.singleton (PartialHomeomorph.refl H)) …
      -/
    · left
      have : e = e' := by
        refine eq_of_eqOnSource_univ (Setoid.symm he'e) ?_ ?_ <;>
          rw [Set.mem_singleton_iff.1 he] <;> rfl
      /-
        case inl.h
        H✝ : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝¹ : TopologicalSpace H✝
        H : Type u
        inst✝ : TopologicalSpace H
        e e' : PartialHomeomorph H H
        he'e : HasEquiv.Equiv e' e
        he : Membership.mem (Singleton.singleton (PartialHomeomorph.refl H)) e
        this : Eq e e'
        ⊢ Membership.mem (Singleton.singleton (PartialHomeomorph.refl H)) e'
      -/
      rwa [← this]
      /-
        🎉 no goals
      -/
      /-
        case inr
        H✝ : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝¹ : TopologicalSpace H✝
        H : Type u
        inst✝ : TopologicalSpace H
        e e' : PartialHomeomorph H H
        he'e : HasEquiv.Equiv e' e
        he : Membership.mem (setOf fun e => Eq e.source EmptyCollection.emptyCollectio …
        ⊢ Membership.mem (Union.union (Singleton.singleton (PartialHomeomorph.refl H)) …
      -/
    · right
      /-
        case inr.h
        H✝ : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝¹ : TopologicalSpace H✝
        H : Type u
        inst✝ : TopologicalSpace H
        e e' : PartialHomeomorph H H
        he'e : HasEquiv.Equiv e' e
        he : Membership.mem (setOf fun e => Eq e.source EmptyCollection.emptyCollectio …
        ⊢ Membership.mem (setOf fun e => Eq e.source EmptyCollection.emptyCollection) e'
      -/
      have he : e.toPartialEquiv.source = ∅ := he
      /-
        case inr.h
        H✝ : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝¹ : TopologicalSpace H✝
        H : Type u
        inst✝ : TopologicalSpace H
        e e' : PartialHomeomorph H H
        he'e : HasEquiv.Equiv e' e
        he✝ : Membership.mem (setOf fun e => Eq e.source EmptyCollection.emptyCollecti …
        he : Eq e.source EmptyCollection.emptyCollection
        ⊢ Membership.mem (setOf fun e => Eq e.source EmptyCollection.emptyCollection) e'
      -/
      rwa [Set.mem_setOf_eq, EqOnSource.source_eq he'e]
      /-
        🎉 no goals
      -/


/-- Every structure groupoid contains the identity groupoid. -/
instance instStructureGroupoidOrderBot : OrderBot (StructureGroupoid H) where
  bot := idGroupoid H
  bot_le := by
    /-
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      ⊢ ∀ (a : StructureGroupoid H), LE.le Bot.bot a
    -/
    intro u f hf
    /-
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      u : StructureGroupoid H
      f : PartialHomeomorph H H
      hf : Membership.mem Bot.bot.members f
      ⊢ Membership.mem u.members f
    -/
    have hf : f ∈ {PartialHomeomorph.refl H} ∪ { e : PartialHomeomorph H H | e.source = ∅ } := hf
    /-
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      u : StructureGroupoid H
      f : PartialHomeomorph H H
      hf✝ : Membership.mem Bot.bot.members f
      hf : Membership.mem (Union.union (Singleton.singleton (PartialHomeomorph.refl  …
      ⊢ Membership.mem u.members f
    -/
    simp only [singleton_union, mem_setOf_eq, mem_insert_iff] at hf
    /-
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      u : StructureGroupoid H
      f : PartialHomeomorph H H
      hf✝ : Membership.mem Bot.bot.members f
      hf : Or (Eq f (PartialHomeomorph.refl H)) (Eq f.source EmptyCollection.emptyCo …
      ⊢ Membership.mem u.members f
    -/
    cases' hf with hf hf
      /-
        case inl
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        u : StructureGroupoid H
        f : PartialHomeomorph H H
        hf✝ : Membership.mem Bot.bot.members f
        hf : Eq f (PartialHomeomorph.refl H)
        ⊢ Membership.mem u.members f
      -/
    · rw [hf]
      /-
        case inl
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        u : StructureGroupoid H
        f : PartialHomeomorph H H
        hf✝ : Membership.mem Bot.bot.members f
        hf : Eq f (PartialHomeomorph.refl H)
        ⊢ Membership.mem u.members (PartialHomeomorph.refl H)
      -/
      apply u.id_mem
      /-
        🎉 no goals
      -/
      /-
        case inr
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        u : StructureGroupoid H
        f : PartialHomeomorph H H
        hf✝ : Membership.mem Bot.bot.members f
        hf : Eq f.source EmptyCollection.emptyCollection
        ⊢ Membership.mem u.members f
      -/
    · apply u.locality
      /-
        case inr
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        u : StructureGroupoid H
        f : PartialHomeomorph H H
        hf✝ : Membership.mem Bot.bot.members f
        hf : Eq f.source EmptyCollection.emptyCollection
        ⊢ ∀ (x : H), Membership.mem f.source x → Exists fun s => And (IsOpen s) (And ( …
      -/
      intro x hx
      /-
        case inr
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        u : StructureGroupoid H
        f : PartialHomeomorph H H
        hf✝ : Membership.mem Bot.bot.members f
        hf : Eq f.source EmptyCollection.emptyCollection
        x : H
        hx : Membership.mem f.source x
        ⊢ Exists fun s => And (IsOpen s) (And (Membership.mem s x) (Membership.mem u ( …
      -/
      rw [hf, mem_empty_iff_false] at hx
      /-
        case inr
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        u : StructureGroupoid H
        f : PartialHomeomorph H H
        hf✝ : Membership.mem Bot.bot.members f
        hf : Eq f.source EmptyCollection.emptyCollection
        x : H
        hx : False
        ⊢ Exists fun s => And (IsOpen s) (And (Membership.mem s x) (Membership.mem u ( …
      -/
      exact hx.elim
      /-
        🎉 no goals
      -/


instance : Inhabited (StructureGroupoid H) := ⟨idGroupoid H⟩


/-- To construct a groupoid, one may consider classes of partial homeomorphisms such that
both the function and its inverse have some property. If this property is stable under composition,
one gets a groupoid. `Pregroupoid` bundles the properties needed for this construction, with the
groupoid of smooth functions with smooth inverses as an application. -/
structure Pregroupoid (H : Type*) [TopologicalSpace H] where
  /-- Property describing membership in this groupoid: the pregroupoid "contains"
    all functions `H → H` having the pregroupoid property on some `s : Set H` -/
  property : (H → H) → Set H → Prop
  /-- The pregroupoid property is stable under composition -/
  comp : ∀ {f g u v}, property f u → property g v →
    IsOpen u → IsOpen v → IsOpen (u ∩ f ⁻¹' v) → property (g ∘ f) (u ∩ f ⁻¹' v)
  /-- Pregroupoids contain the identity map (on `univ`) -/
  id_mem : property id univ
  /-- The pregroupoid property is "local", in the sense that `f` has the pregroupoid property on `u`
  iff its restriction to each open subset of `u` has it -/
  locality :
    ∀ {f u}, IsOpen u → (∀ x ∈ u, ∃ v, IsOpen v ∧ x ∈ v ∧ property f (u ∩ v)) → property f u
  /-- If `f = g` on `u` and `property f u`, then `property g u` -/
  congr : ∀ {f g : H → H} {u}, IsOpen u → (∀ x ∈ u, g x = f x) → property f u → property g u


/-- Construct a groupoid of partial homeos for which the map and its inverse have some property,
from a pregroupoid asserting that this property is stable under composition. -/
def Pregroupoid.groupoid (PG : Pregroupoid H) : StructureGroupoid H where
  members := { e : PartialHomeomorph H H | PG.property e e.source ∧ PG.property e.symm e.target }
  trans' e e' he he' := by
    /-
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      PG : Pregroupoid H
      e e' : PartialHomeomorph H H
      he : Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.proper …
      he' : Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.prope …
      ⊢ Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.property  …
    -/
    constructor
      /-
        case left
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        PG : Pregroupoid H
        e e' : PartialHomeomorph H H
        he : Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.proper …
        he' : Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.prope …
        ⊢ PG.property (↑(e.trans e')) (e.trans e').source
      -/
    · apply PG.comp he.1 he'.1 e.open_source e'.open_source
      /-
        case left
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        PG : Pregroupoid H
        e e' : PartialHomeomorph H H
        he : Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.proper …
        he' : Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.prope …
        ⊢ IsOpen (Inter.inter e.source (Set.preimage (↑e) e'.source))
      -/
      apply e.continuousOn_toFun.isOpen_inter_preimage e.open_source e'.open_source
      /-
        🎉 no goals
      -/
      /-
        case right
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        PG : Pregroupoid H
        e e' : PartialHomeomorph H H
        he : Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.proper …
        he' : Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.prope …
        ⊢ PG.property (↑(e.trans e').symm) (e.trans e').target
      -/
    · apply PG.comp he'.2 he.2 e'.open_target e.open_target
      /-
        case right
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        PG : Pregroupoid H
        e e' : PartialHomeomorph H H
        he : Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.proper …
        he' : Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.prope …
        ⊢ IsOpen (Inter.inter e'.target (Set.preimage (↑e'.symm) e.target))
      -/
      apply e'.continuousOn_invFun.isOpen_inter_preimage e'.open_target e.open_target
      /-
        🎉 no goals
      -/
  symm' _ he := ⟨he.2, he.1⟩
  id_mem' := ⟨PG.id_mem, PG.id_mem⟩
  locality' e he := by
    /-
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      PG : Pregroupoid H
      e : PartialHomeomorph H H
      he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
      ⊢ Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.property  …
    -/
    constructor
      /-
        case left
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        PG : Pregroupoid H
        e : PartialHomeomorph H H
        he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
        ⊢ PG.property (↑e) e.source
      -/
    · refine PG.locality e.open_source fun x xu ↦ ?_
      /-
        case left
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        PG : Pregroupoid H
        e : PartialHomeomorph H H
        he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
        x : H
        xu : Membership.mem e.source x
        ⊢ Exists fun v => And (IsOpen v) (And (Membership.mem v x) (PG.property (↑e) ( …
      -/
      rcases he x xu with ⟨s, s_open, xs, hs⟩
      /-
        case left.intro.intro.intro
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        PG : Pregroupoid H
        e : PartialHomeomorph H H
        he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
        x : H
        xu : Membership.mem e.source x
        s : Set H
        s_open : IsOpen s
        xs : Membership.mem s x
        hs : Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.proper …
        ⊢ Exists fun v => And (IsOpen v) (And (Membership.mem v x) (PG.property (↑e) ( …
      -/
      refine ⟨s, s_open, xs, ?_⟩
      /-
        case left.intro.intro.intro
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        PG : Pregroupoid H
        e : PartialHomeomorph H H
        he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
        x : H
        xu : Membership.mem e.source x
        s : Set H
        s_open : IsOpen s
        xs : Membership.mem s x
        hs : Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.proper …
        ⊢ PG.property (↑e) (Inter.inter e.source s)
      -/
      convert hs.1 using 1
      /-
        case h.e'_5
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        PG : Pregroupoid H
        e : PartialHomeomorph H H
        he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
        x : H
        xu : Membership.mem e.source x
        s : Set H
        s_open : IsOpen s
        xs : Membership.mem s x
        hs : Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.proper …
        ⊢ Eq (Inter.inter e.source s) (e.restr s).source
      -/
      dsimp [PartialHomeomorph.restr]
      /-
        case h.e'_5
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        PG : Pregroupoid H
        e : PartialHomeomorph H H
        he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
        x : H
        xu : Membership.mem e.source x
        s : Set H
        s_open : IsOpen s
        xs : Membership.mem s x
        hs : Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.proper …
        ⊢ Eq (Inter.inter e.source s) (Inter.inter e.source (interior s))
      -/
      rw [s_open.interior_eq]
      /-
        🎉 no goals
      -/
      /-
        case right
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        PG : Pregroupoid H
        e : PartialHomeomorph H H
        he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
        ⊢ PG.property (↑e.symm) e.target
      -/
    · refine PG.locality e.open_target fun x xu ↦ ?_
      /-
        case right
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        PG : Pregroupoid H
        e : PartialHomeomorph H H
        he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
        x : H
        xu : Membership.mem e.target x
        ⊢ Exists fun v => And (IsOpen v) (And (Membership.mem v x) (PG.property (↑e.sy …
      -/
      rcases he (e.symm x) (e.map_target xu) with ⟨s, s_open, xs, hs⟩
      /-
        case right.intro.intro.intro
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        PG : Pregroupoid H
        e : PartialHomeomorph H H
        he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
        x : H
        xu : Membership.mem e.target x
        s : Set H
        s_open : IsOpen s
        xs : Membership.mem s (↑e.symm x)
        hs : Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.proper …
        ⊢ Exists fun v => And (IsOpen v) (And (Membership.mem v x) (PG.property (↑e.sy …
      -/
      refine ⟨e.target ∩ e.symm ⁻¹' s, ?_, ⟨xu, xs⟩, ?_⟩
        /-
          case right.intro.intro.intro.refine_1
          H : Type u
          H' : Type u_1
          M : Type u_2
          M' : Type u_3
          M'' : Type u_4
          inst✝ : TopologicalSpace H
          PG : Pregroupoid H
          e : PartialHomeomorph H H
          he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
          x : H
          xu : Membership.mem e.target x
          s : Set H
          s_open : IsOpen s
          xs : Membership.mem s (↑e.symm x)
          hs : Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.proper …
          ⊢ IsOpen (Inter.inter e.target (Set.preimage (↑e.symm) s))
        -/
      · exact ContinuousOn.isOpen_inter_preimage e.continuousOn_invFun e.open_target s_open
        /-
          🎉 no goals
        -/
        /-
          case right.intro.intro.intro.refine_2
          H : Type u
          H' : Type u_1
          M : Type u_2
          M' : Type u_3
          M'' : Type u_4
          inst✝ : TopologicalSpace H
          PG : Pregroupoid H
          e : PartialHomeomorph H H
          he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
          x : H
          xu : Membership.mem e.target x
          s : Set H
          s_open : IsOpen s
          xs : Membership.mem s (↑e.symm x)
          hs : Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.proper …
          ⊢ PG.property (↑e.symm) (Inter.inter e.target (Inter.inter e.target (Set.preim …
        -/
      · rw [← inter_assoc, inter_self]
        /-
          case right.intro.intro.intro.refine_2
          H : Type u
          H' : Type u_1
          M : Type u_2
          M' : Type u_3
          M'' : Type u_4
          inst✝ : TopologicalSpace H
          PG : Pregroupoid H
          e : PartialHomeomorph H H
          he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
          x : H
          xu : Membership.mem e.target x
          s : Set H
          s_open : IsOpen s
          xs : Membership.mem s (↑e.symm x)
          hs : Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.proper …
          ⊢ PG.property (↑e.symm) (Inter.inter e.target (Set.preimage (↑e.symm) s))
        -/
        convert hs.2 using 1
        /-
          case h.e'_5
          H : Type u
          H' : Type u_1
          M : Type u_2
          M' : Type u_3
          M'' : Type u_4
          inst✝ : TopologicalSpace H
          PG : Pregroupoid H
          e : PartialHomeomorph H H
          he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
          x : H
          xu : Membership.mem e.target x
          s : Set H
          s_open : IsOpen s
          xs : Membership.mem s (↑e.symm x)
          hs : Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.proper …
          ⊢ Eq (Inter.inter e.target (Set.preimage (↑e.symm) s)) (e.restr s).target
        -/
        dsimp [PartialHomeomorph.restr]
        /-
          case h.e'_5
          H : Type u
          H' : Type u_1
          M : Type u_2
          M' : Type u_3
          M'' : Type u_4
          inst✝ : TopologicalSpace H
          PG : Pregroupoid H
          e : PartialHomeomorph H H
          he : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (An …
          x : H
          xu : Membership.mem e.target x
          s : Set H
          s_open : IsOpen s
          xs : Membership.mem s (↑e.symm x)
          hs : Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.proper …
          ⊢ Eq (Inter.inter e.target (Set.preimage (↑e.symm) s)) (Inter.inter e.target ( …
        -/
        rw [s_open.interior_eq]
        /-
          🎉 no goals
        -/
  mem_of_eqOnSource' e e' he ee' := by
    /-
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      PG : Pregroupoid H
      e e' : PartialHomeomorph H H
      he : Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.proper …
      ee' : HasEquiv.Equiv e' e
      ⊢ Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.property  …
    -/
    constructor
      /-
        case left
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        PG : Pregroupoid H
        e e' : PartialHomeomorph H H
        he : Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.proper …
        ee' : HasEquiv.Equiv e' e
        ⊢ PG.property (↑e') e'.source
      -/
    · apply PG.congr e'.open_source ee'.2
      /-
        case left
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        PG : Pregroupoid H
        e e' : PartialHomeomorph H H
        he : Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.proper …
        ee' : HasEquiv.Equiv e' e
        ⊢ PG.property (↑e) e'.source
      -/
      simp only [ee'.1, he.1]
      /-
        🎉 no goals
      -/
      /-
        case right
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        PG : Pregroupoid H
        e e' : PartialHomeomorph H H
        he : Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.proper …
        ee' : HasEquiv.Equiv e' e
        ⊢ PG.property (↑e'.symm) e'.target
      -/
    · have A := EqOnSource.symm' ee'
      /-
        case right
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        PG : Pregroupoid H
        e e' : PartialHomeomorph H H
        he : Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.proper …
        ee' : HasEquiv.Equiv e' e
        A : HasEquiv.Equiv e'.symm e.symm
        ⊢ PG.property (↑e'.symm) e'.target
      -/
      apply PG.congr e'.symm.open_source A.2
      -- Porting note: was
      -- convert he.2
      -- rw [A.1]
      -- rfl
      /-
        case right
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        PG : Pregroupoid H
        e e' : PartialHomeomorph H H
        he : Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.proper …
        ee' : HasEquiv.Equiv e' e
        A : HasEquiv.Equiv e'.symm e.symm
        ⊢ PG.property (↑e.symm) e'.symm.source
      -/
      rw [A.1, symm_toPartialEquiv, PartialEquiv.symm_source]
      /-
        case right
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        PG : Pregroupoid H
        e e' : PartialHomeomorph H H
        he : Membership.mem (setOf fun e => And (PG.property (↑e) e.source) (PG.proper …
        ee' : HasEquiv.Equiv e' e
        A : HasEquiv.Equiv e'.symm e.symm
        ⊢ PG.property (↑e.symm) e.target
      -/
      exact he.2
      /-
        🎉 no goals
      -/


theorem mem_groupoid_of_pregroupoid {PG : Pregroupoid H} {e : PartialHomeomorph H H} :
    e ∈ PG.groupoid ↔ PG.property e e.source ∧ PG.property e.symm e.target :=
  Iff.rfl


theorem groupoid_of_pregroupoid_le (PG₁ PG₂ : Pregroupoid H)
    (h : ∀ f s, PG₁.property f s → PG₂.property f s) : PG₁.groupoid ≤ PG₂.groupoid := by
  /-
    H : Type u
    inst✝ : TopologicalSpace H
    PG₁ PG₂ : Pregroupoid H
    h : ∀ (f : H → H) (s : Set H), PG₁.property f s → PG₂.property f s
    ⊢ LE.le PG₁.groupoid PG₂.groupoid
  -/
  refine StructureGroupoid.le_iff.2 fun e he ↦ ?_
  /-
    H : Type u
    inst✝ : TopologicalSpace H
    PG₁ PG₂ : Pregroupoid H
    h : ∀ (f : H → H) (s : Set H), PG₁.property f s → PG₂.property f s
    e : PartialHomeomorph H H
    he : Membership.mem PG₁.groupoid e
    ⊢ Membership.mem PG₂.groupoid e
  -/
  rw [mem_groupoid_of_pregroupoid] at he ⊢
  /-
    H : Type u
    inst✝ : TopologicalSpace H
    PG₁ PG₂ : Pregroupoid H
    h : ∀ (f : H → H) (s : Set H), PG₁.property f s → PG₂.property f s
    e : PartialHomeomorph H H
    he : And (PG₁.property (↑e) e.source) (PG₁.property (↑e.symm) e.target)
    ⊢ And (PG₂.property (↑e) e.source) (PG₂.property (↑e.symm) e.target)
  -/
  exact ⟨h _ _ he.1, h _ _ he.2⟩
  /-
    🎉 no goals
  -/


theorem mem_pregroupoid_of_eqOnSource (PG : Pregroupoid H) {e e' : PartialHomeomorph H H}
    (he' : e ≈ e') (he : PG.property e e.source) : PG.property e' e'.source := by
  /-
    H : Type u
    inst✝ : TopologicalSpace H
    PG : Pregroupoid H
    e e' : PartialHomeomorph H H
    he' : HasEquiv.Equiv e e'
    he : PG.property (↑e) e.source
    ⊢ PG.property (↑e') e'.source
  -/
  rw [← he'.1]
  /-
    H : Type u
    inst✝ : TopologicalSpace H
    PG : Pregroupoid H
    e e' : PartialHomeomorph H H
    he' : HasEquiv.Equiv e e'
    he : PG.property (↑e) e.source
    ⊢ PG.property (↑e') e.source
  -/
  exact PG.congr e.open_source he'.eqOn.symm he
  /-
    🎉 no goals
  -/


/-- The pregroupoid of all partial maps on a topological space `H`. -/
abbrev continuousPregroupoid (H : Type*) [TopologicalSpace H] : Pregroupoid H where
  property _ _ := True
  comp _ _ _ _ _ := trivial
  id_mem := trivial
  locality _ _ := trivial
  congr _ _ _ := trivial


instance (H : Type*) [TopologicalSpace H] : Inhabited (Pregroupoid H) :=
  ⟨continuousPregroupoid H⟩


/-- The groupoid of all partial homeomorphisms on a topological space `H`. -/
def continuousGroupoid (H : Type*) [TopologicalSpace H] : StructureGroupoid H :=
  Pregroupoid.groupoid (continuousPregroupoid H)


/-- Every structure groupoid is contained in the groupoid of all partial homeomorphisms. -/
instance instStructureGroupoidOrderTop : OrderTop (StructureGroupoid H) where
  top := continuousGroupoid H
  le_top _ _ _ := ⟨trivial, trivial⟩


instance : CompleteLattice (StructureGroupoid H) :=
  { SetLike.instPartialOrder,
    completeLatticeOfInf _ (by
      exact fun s =>
      ⟨fun S Ss F hF => mem_iInter₂.mp hF S Ss,
      fun T Tl F fT => mem_iInter₂.mpr (fun i his => Tl his fT)⟩) with
    le := (· ≤ ·)
    lt := (· < ·)
    bot := instStructureGroupoidOrderBot.bot
    bot_le := instStructureGroupoidOrderBot.bot_le
    top := instStructureGroupoidOrderTop.top
    le_top := instStructureGroupoidOrderTop.le_top
    inf := (· ⊓ ·)
    le_inf := fun _ _ _ h₁₂ h₁₃ _ hm ↦ ⟨h₁₂ hm, h₁₃ hm⟩
    inf_le_left := fun _ _ _ ↦ And.left
    inf_le_right := fun _ _ _ ↦ And.right }


/-- A groupoid is closed under restriction if it contains all restrictions of its element local
homeomorphisms to open subsets of the source. -/
class ClosedUnderRestriction (G : StructureGroupoid H) : Prop where
  closedUnderRestriction :
    ∀ {e : PartialHomeomorph H H}, e ∈ G → ∀ s : Set H, IsOpen s → e.restr s ∈ G


theorem closedUnderRestriction' {G : StructureGroupoid H} [ClosedUnderRestriction G]
    {e : PartialHomeomorph H H} (he : e ∈ G) {s : Set H} (hs : IsOpen s) : e.restr s ∈ G :=
  ClosedUnderRestriction.closedUnderRestriction he s hs


/-- The trivial restriction-closed groupoid, containing only partial homeomorphisms equivalent
to the restriction of the identity to the various open subsets. -/
def idRestrGroupoid : StructureGroupoid H where
  members := { e | ∃ (s : Set H) (h : IsOpen s), e ≈ PartialHomeomorph.ofSet s h }
  trans' := by
    /-
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      ⊢ ∀ (e e' : PartialHomeomorph H H), Membership.mem (setOf fun e => Exists fun  …
    -/
    rintro e e' ⟨s, hs, hse⟩ ⟨s', hs', hse'⟩
    /-
      case intro.intro.intro.intro
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      e e' : PartialHomeomorph H H
      s : Set H
      hs : IsOpen s
      hse : HasEquiv.Equiv e (PartialHomeomorph.ofSet s hs)
      s' : Set H
      hs' : IsOpen s'
      hse' : HasEquiv.Equiv e' (PartialHomeomorph.ofSet s' hs')
      ⊢ Membership.mem (setOf fun e => Exists fun s => Exists fun h => HasEquiv.Equi …
    -/
    refine ⟨s ∩ s', hs.inter hs', ?_⟩
    /-
      case intro.intro.intro.intro
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      e e' : PartialHomeomorph H H
      s : Set H
      hs : IsOpen s
      hse : HasEquiv.Equiv e (PartialHomeomorph.ofSet s hs)
      s' : Set H
      hs' : IsOpen s'
      hse' : HasEquiv.Equiv e' (PartialHomeomorph.ofSet s' hs')
      ⊢ HasEquiv.Equiv (e.trans e') (PartialHomeomorph.ofSet (Inter.inter s s') ⋯)
    -/
    have := PartialHomeomorph.EqOnSource.trans' hse hse'
    /-
      case intro.intro.intro.intro
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      e e' : PartialHomeomorph H H
      s : Set H
      hs : IsOpen s
      hse : HasEquiv.Equiv e (PartialHomeomorph.ofSet s hs)
      s' : Set H
      hs' : IsOpen s'
      hse' : HasEquiv.Equiv e' (PartialHomeomorph.ofSet s' hs')
      this : HasEquiv.Equiv (e.trans e') ((PartialHomeomorph.ofSet s hs).trans (Part …
      ⊢ HasEquiv.Equiv (e.trans e') (PartialHomeomorph.ofSet (Inter.inter s s') ⋯)
    -/
    rwa [PartialHomeomorph.ofSet_trans_ofSet] at this
    /-
      🎉 no goals
    -/
  symm' := by
    /-
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      ⊢ ∀ (e : PartialHomeomorph H H), Membership.mem (setOf fun e => Exists fun s = …
    -/
    rintro e ⟨s, hs, hse⟩
    /-
      case intro.intro
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      e : PartialHomeomorph H H
      s : Set H
      hs : IsOpen s
      hse : HasEquiv.Equiv e (PartialHomeomorph.ofSet s hs)
      ⊢ Membership.mem (setOf fun e => Exists fun s => Exists fun h => HasEquiv.Equi …
    -/
    refine ⟨s, hs, ?_⟩
    /-
      case intro.intro
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      e : PartialHomeomorph H H
      s : Set H
      hs : IsOpen s
      hse : HasEquiv.Equiv e (PartialHomeomorph.ofSet s hs)
      ⊢ HasEquiv.Equiv e.symm (PartialHomeomorph.ofSet s hs)
    -/
    rw [← ofSet_symm]
    /-
      case intro.intro
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      e : PartialHomeomorph H H
      s : Set H
      hs : IsOpen s
      hse : HasEquiv.Equiv e (PartialHomeomorph.ofSet s hs)
      ⊢ HasEquiv.Equiv e.symm (PartialHomeomorph.ofSet s hs).symm
    -/
    exact PartialHomeomorph.EqOnSource.symm' hse
    /-
      🎉 no goals
    -/
                                    /-
                                      H : Type u
                                      H' : Type u_1
                                      M : Type u_2
                                      M' : Type u_3
                                      M'' : Type u_4
                                      inst✝ : TopologicalSpace H
                                      ⊢ HasEquiv.Equiv (PartialHomeomorph.refl H) (PartialHomeomorph.ofSet Set.univ ⋯)
                                    -/
  id_mem' := ⟨univ, isOpen_univ, by simp only [mfld_simps, refl]⟩
                                    /-
                                      🎉 no goals
                                    -/
  locality' := by
    /-
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      ⊢ ∀ (e : PartialHomeomorph H H), (∀ (x : H), Membership.mem e.source x → Exist …
    -/
    intro e h
    /-
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      e : PartialHomeomorph H H
      h : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (And …
      ⊢ Membership.mem (setOf fun e => Exists fun s => Exists fun h => HasEquiv.Equi …
    -/
    refine ⟨e.source, e.open_source, by simp only [mfld_simps], ?_⟩
    /-
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      e : PartialHomeomorph H H
      h : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (And …
      ⊢ Set.EqOn (↑e) (↑(PartialHomeomorph.ofSet e.source ⋯)) e.source
    -/
    intro x hx
    /-
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      e : PartialHomeomorph H H
      h : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (And …
      x : H
      hx : Membership.mem e.source x
      ⊢ Eq (↑e x) (↑(PartialHomeomorph.ofSet e.source ⋯) x)
    -/
    rcases h x hx with ⟨s, hs, hxs, s', hs', hes'⟩
    have hes : x ∈ (e.restr s).source := by
      rw [e.restr_source]
      refine ⟨hx, ?_⟩
      rw [hs.interior_eq]
      exact hxs
    /-
      case intro.intro.intro.intro.intro
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      e : PartialHomeomorph H H
      h : ∀ (x : H), Membership.mem e.source x → Exists fun s => And (IsOpen s) (And …
      x : H
      hx : Membership.mem e.source x
      s : Set H
      hs : IsOpen s
      hxs : Membership.mem s x
      s' : Set H
      hs' : IsOpen s'
      hes' : HasEquiv.Equiv (e.restr s) (PartialHomeomorph.ofSet s' hs')
      hes : Membership.mem (e.restr s).source x
      ⊢ Eq (↑e x) (↑(PartialHomeomorph.ofSet e.source ⋯) x)
    -/
    simpa only [mfld_simps] using PartialHomeomorph.EqOnSource.eqOn hes' hes
    /-
      🎉 no goals
    -/
  mem_of_eqOnSource' := by
    /-
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      ⊢ ∀ (e e' : PartialHomeomorph H H), Membership.mem (setOf fun e => Exists fun  …
    -/
    rintro e e' ⟨s, hs, hse⟩ hee'
    /-
      case intro.intro
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      e e' : PartialHomeomorph H H
      s : Set H
      hs : IsOpen s
      hse : HasEquiv.Equiv e (PartialHomeomorph.ofSet s hs)
      hee' : HasEquiv.Equiv e' e
      ⊢ Membership.mem (setOf fun e => Exists fun s => Exists fun h => HasEquiv.Equi …
    -/
    exact ⟨s, hs, Setoid.trans hee' hse⟩
    /-
      🎉 no goals
    -/


theorem idRestrGroupoid_mem {s : Set H} (hs : IsOpen s) : ofSet s hs ∈ @idRestrGroupoid H _ :=
  ⟨s, hs, refl _⟩


/-- The trivial restriction-closed groupoid is indeed `ClosedUnderRestriction`. -/
instance closedUnderRestriction_idRestrGroupoid : ClosedUnderRestriction (@idRestrGroupoid H _) :=
  ⟨by
    /-
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      ⊢ ∀ {e : PartialHomeomorph H H}, Membership.mem idRestrGroupoid e → ∀ (s : Set …
    -/
    rintro e ⟨s', hs', he⟩ s hs
    /-
      case intro.intro
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      e : PartialHomeomorph H H
      s' : Set H
      hs' : IsOpen s'
      he : HasEquiv.Equiv e (PartialHomeomorph.ofSet s' hs')
      s : Set H
      hs : IsOpen s
      ⊢ Membership.mem idRestrGroupoid (e.restr s)
    -/
    use s' ∩ s, hs'.inter hs
    /-
      case h
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      e : PartialHomeomorph H H
      s' : Set H
      hs' : IsOpen s'
      he : HasEquiv.Equiv e (PartialHomeomorph.ofSet s' hs')
      s : Set H
      hs : IsOpen s
      ⊢ HasEquiv.Equiv (e.restr s) (PartialHomeomorph.ofSet (Inter.inter s' s) ⋯)
    -/
    refine Setoid.trans (PartialHomeomorph.EqOnSource.restr he s) ?_
    /-
      case h
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝ : TopologicalSpace H
      e : PartialHomeomorph H H
      s' : Set H
      hs' : IsOpen s'
      he : HasEquiv.Equiv e (PartialHomeomorph.ofSet s' hs')
      s : Set H
      hs : IsOpen s
      ⊢ HasEquiv.Equiv ((PartialHomeomorph.ofSet s' hs').restr s) (PartialHomeomorph …
    -/
    exact ⟨by simp only [hs.interior_eq, mfld_simps], by simp only [mfld_simps, eqOn_refl]⟩⟩
    /-
      🎉 no goals
    -/


/-- A groupoid is closed under restriction if and only if it contains the trivial restriction-closed
groupoid. -/
theorem closedUnderRestriction_iff_id_le (G : StructureGroupoid H) :
    ClosedUnderRestriction G ↔ idRestrGroupoid ≤ G := by
  /-
    H : Type u
    inst✝ : TopologicalSpace H
    G : StructureGroupoid H
    ⊢ Iff (ClosedUnderRestriction G) (LE.le idRestrGroupoid G)
  -/
  constructor
    /-
      case mp
      H : Type u
      inst✝ : TopologicalSpace H
      G : StructureGroupoid H
      ⊢ ClosedUnderRestriction G → LE.le idRestrGroupoid G
    -/
  · intro _i
    /-
      case mp
      H : Type u
      inst✝ : TopologicalSpace H
      G : StructureGroupoid H
      _i : ClosedUnderRestriction G
      ⊢ LE.le idRestrGroupoid G
    -/
    rw [StructureGroupoid.le_iff]
    /-
      case mp
      H : Type u
      inst✝ : TopologicalSpace H
      G : StructureGroupoid H
      _i : ClosedUnderRestriction G
      ⊢ ∀ (e : PartialHomeomorph H H), Membership.mem idRestrGroupoid e → Membership …
    -/
    rintro e ⟨s, hs, hes⟩
    /-
      case mp.intro.intro
      H : Type u
      inst✝ : TopologicalSpace H
      G : StructureGroupoid H
      _i : ClosedUnderRestriction G
      e : PartialHomeomorph H H
      s : Set H
      hs : IsOpen s
      hes : HasEquiv.Equiv e (PartialHomeomorph.ofSet s hs)
      ⊢ Membership.mem G e
    -/
    refine G.mem_of_eqOnSource ?_ hes
    /-
      case mp.intro.intro
      H : Type u
      inst✝ : TopologicalSpace H
      G : StructureGroupoid H
      _i : ClosedUnderRestriction G
      e : PartialHomeomorph H H
      s : Set H
      hs : IsOpen s
      hes : HasEquiv.Equiv e (PartialHomeomorph.ofSet s hs)
      ⊢ Membership.mem G (PartialHomeomorph.ofSet s hs)
    -/
    convert closedUnderRestriction' G.id_mem hs
    -- Porting note: was
    -- change s = _ ∩ _
    -- rw [hs.interior_eq]
    -- simp only [mfld_simps]
    /-
      case h.e'_5
      H : Type u
      inst✝ : TopologicalSpace H
      G : StructureGroupoid H
      _i : ClosedUnderRestriction G
      e : PartialHomeomorph H H
      s : Set H
      hs : IsOpen s
      hes : HasEquiv.Equiv e (PartialHomeomorph.ofSet s hs)
      ⊢ Eq (PartialHomeomorph.ofSet s hs) ((PartialHomeomorph.refl H).restr s)
    -/
    ext
      /-
        case h.e'_5.h
        H : Type u
        inst✝ : TopologicalSpace H
        G : StructureGroupoid H
        _i : ClosedUnderRestriction G
        e : PartialHomeomorph H H
        s : Set H
        hs : IsOpen s
        hes : HasEquiv.Equiv e (PartialHomeomorph.ofSet s hs)
        x✝ : H
        ⊢ Eq (↑(PartialHomeomorph.ofSet s hs) x✝) (↑((PartialHomeomorph.refl H).restr  …
      -/
    · rw [PartialHomeomorph.restr_apply, PartialHomeomorph.refl_apply, id, ofSet_apply, id_eq]
      /-
        🎉 no goals
      -/
      /-
        case h.e'_5.hinv
        H : Type u
        inst✝ : TopologicalSpace H
        G : StructureGroupoid H
        _i : ClosedUnderRestriction G
        e : PartialHomeomorph H H
        s : Set H
        hs : IsOpen s
        hes : HasEquiv.Equiv e (PartialHomeomorph.ofSet s hs)
        x✝ : H
        ⊢ Eq (↑(PartialHomeomorph.ofSet s hs).symm x✝) (↑((PartialHomeomorph.refl H).r …
      -/
    · simp [hs]
      /-
        🎉 no goals
      -/
      /-
        case h.e'_5.hs.h
        H : Type u
        inst✝ : TopologicalSpace H
        G : StructureGroupoid H
        _i : ClosedUnderRestriction G
        e : PartialHomeomorph H H
        s : Set H
        hs : IsOpen s
        hes : HasEquiv.Equiv e (PartialHomeomorph.ofSet s hs)
        x✝ : H
        ⊢ Iff (Membership.mem (PartialHomeomorph.ofSet s hs).source x✝) (Membership.me …
      -/
    · simp [hs.interior_eq]
      /-
        🎉 no goals
      -/
    /-
      case mpr
      H : Type u
      inst✝ : TopologicalSpace H
      G : StructureGroupoid H
      ⊢ LE.le idRestrGroupoid G → ClosedUnderRestriction G
    -/
  · intro h
    /-
      case mpr
      H : Type u
      inst✝ : TopologicalSpace H
      G : StructureGroupoid H
      h : LE.le idRestrGroupoid G
      ⊢ ClosedUnderRestriction G
    -/
    constructor
    /-
      case mpr.closedUnderRestriction
      H : Type u
      inst✝ : TopologicalSpace H
      G : StructureGroupoid H
      h : LE.le idRestrGroupoid G
      ⊢ ∀ {e : PartialHomeomorph H H}, Membership.mem G e → ∀ (s : Set H), IsOpen s  …
    -/
    intro e he s hs
    /-
      case mpr.closedUnderRestriction
      H : Type u
      inst✝ : TopologicalSpace H
      G : StructureGroupoid H
      h : LE.le idRestrGroupoid G
      e : PartialHomeomorph H H
      he : Membership.mem G e
      s : Set H
      hs : IsOpen s
      ⊢ Membership.mem G (e.restr s)
    -/
    rw [← ofSet_trans (e : PartialHomeomorph H H) hs]
    /-
      case mpr.closedUnderRestriction
      H : Type u
      inst✝ : TopologicalSpace H
      G : StructureGroupoid H
      h : LE.le idRestrGroupoid G
      e : PartialHomeomorph H H
      he : Membership.mem G e
      s : Set H
      hs : IsOpen s
      ⊢ Membership.mem G ((PartialHomeomorph.ofSet s hs).trans e)
    -/
    refine G.trans ?_ he
    /-
      case mpr.closedUnderRestriction
      H : Type u
      inst✝ : TopologicalSpace H
      G : StructureGroupoid H
      h : LE.le idRestrGroupoid G
      e : PartialHomeomorph H H
      he : Membership.mem G e
      s : Set H
      hs : IsOpen s
      ⊢ Membership.mem G (PartialHomeomorph.ofSet s hs)
    -/
    apply StructureGroupoid.le_iff.mp h
    /-
      case mpr.closedUnderRestriction.a
      H : Type u
      inst✝ : TopologicalSpace H
      G : StructureGroupoid H
      h : LE.le idRestrGroupoid G
      e : PartialHomeomorph H H
      he : Membership.mem G e
      s : Set H
      hs : IsOpen s
      ⊢ Membership.mem idRestrGroupoid (PartialHomeomorph.ofSet s hs)
    -/
    exact idRestrGroupoid_mem hs
    /-
      🎉 no goals
    -/


/-- The groupoid of all partial homeomorphisms on a topological space `H`
is closed under restriction. -/
instance : ClosedUnderRestriction (continuousGroupoid H) :=
  (closedUnderRestriction_iff_id_le _).mpr le_top


/-- A charted space is a topological space endowed with an atlas, i.e., a set of local
homeomorphisms taking value in a model space `H`, called charts, such that the domains of the charts
cover the whole space. We express the covering property by choosing for each `x` a member
`chartAt x` of the atlas containing `x` in its source: in the smooth case, this is convenient to
construct the tangent bundle in an efficient way.
The model space is written as an explicit parameter as there can be several model spaces for a
given topological space. For instance, a complex manifold (modelled over `ℂ^n`) will also be seen
sometimes as a real manifold over `ℝ^(2n)`.
-/
@[ext]
class ChartedSpace (H : Type*) [TopologicalSpace H] (M : Type*) [TopologicalSpace M] where
  /-- The atlas of charts in the `ChartedSpace`. -/
  protected atlas : Set (PartialHomeomorph M H)
  /-- The preferred chart at each point in the charted space. -/
  protected chartAt : M → PartialHomeomorph M H
  protected mem_chart_source : ∀ x, x ∈ (chartAt x).source
  protected chart_mem_atlas : ∀ x, chartAt x ∈ atlas


/-- The atlas of charts in a `ChartedSpace`. -/
abbrev atlas (H : Type*) [TopologicalSpace H] (M : Type*) [TopologicalSpace M]
    [ChartedSpace H M] : Set (PartialHomeomorph M H) :=
  ChartedSpace.atlas


/-- The preferred chart at a point `x` in a charted space `M`. -/
abbrev chartAt (H : Type*) [TopologicalSpace H] {M : Type*} [TopologicalSpace M]
    [ChartedSpace H M] (x : M) : PartialHomeomorph M H :=
  ChartedSpace.chartAt x


@[simp, mfld_simps]
lemma mem_chart_source (H : Type*) {M : Type*} [TopologicalSpace H] [TopologicalSpace M]
    [ChartedSpace H M] (x : M) : x ∈ (chartAt H x).source :=
  ChartedSpace.mem_chart_source x


@[simp, mfld_simps]
lemma chart_mem_atlas (H : Type*) {M : Type*} [TopologicalSpace H] [TopologicalSpace M]
    [ChartedSpace H M] (x : M) : chartAt H x ∈ atlas H M :=
  ChartedSpace.chart_mem_atlas x


/-- An empty type is a charted space over any topological space. -/
def ChartedSpace.empty (H : Type*) [TopologicalSpace H]
    (M : Type*) [TopologicalSpace M] [IsEmpty M] : ChartedSpace H M where
  atlas := ∅
  chartAt x := (IsEmpty.false x).elim
  mem_chart_source x := (IsEmpty.false x).elim
  chart_mem_atlas x := (IsEmpty.false x).elim


/-- Any space is a `ChartedSpace` modelled over itself, by just using the identity chart. -/
instance chartedSpaceSelf (H : Type*) [TopologicalSpace H] : ChartedSpace H H where
  atlas := {PartialHomeomorph.refl H}
  chartAt _ := PartialHomeomorph.refl H
  mem_chart_source x := mem_univ x
  chart_mem_atlas _ := mem_singleton _


/-- In the trivial `ChartedSpace` structure of a space modelled over itself through the identity,
the atlas members are just the identity. -/
@[simp, mfld_simps]
theorem chartedSpaceSelf_atlas {H : Type*} [TopologicalSpace H] {e : PartialHomeomorph H H} :
    e ∈ atlas H H ↔ e = PartialHomeomorph.refl H :=
  Iff.rfl


/-- In the model space, `chartAt` is always the identity. -/
theorem chartAt_self_eq {H : Type*} [TopologicalSpace H] {x : H} :
    chartAt H x = PartialHomeomorph.refl H := rfl


theorem mem_chart_target (x : M) : chartAt H x x ∈ (chartAt H x).target :=
  (chartAt H x).map_source (mem_chart_source _ _)


theorem chart_source_mem_nhds (x : M) : (chartAt H x).source ∈ 𝓝 x :=
  (chartAt H x).open_source.mem_nhds <| mem_chart_source H x


theorem chart_target_mem_nhds (x : M) : (chartAt H x).target ∈ 𝓝 (chartAt H x x) :=
  (chartAt H x).open_target.mem_nhds <| mem_chart_target H x


variable (M) in
@[simp]
theorem iUnion_source_chartAt : (⋃ x : M, (chartAt H x).source) = (univ : Set M) :=
  eq_univ_iff_forall.mpr fun x ↦ mem_iUnion.mpr ⟨x, mem_chart_source H x⟩


theorem ChartedSpace.isOpen_iff (s : Set M) :
    IsOpen s ↔ ∀ x : M, IsOpen <| chartAt H x '' ((chartAt H x).source ∩ s) := by
  /-
    H : Type u
    M : Type u_2
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    s : Set M
    ⊢ Iff (IsOpen s) (∀ (x : M), IsOpen (Set.image (↑(chartAt H x)) (Inter.inter ( …
  -/
  rw [isOpen_iff_of_cover (fun i ↦ (chartAt H i).open_source) (iUnion_source_chartAt H M)]
  /-
    H : Type u
    M : Type u_2
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    s : Set M
    ⊢ Iff (∀ (i : M), IsOpen (Inter.inter (chartAt H i).source s)) (∀ (x : M), IsO …
  -/
  simp only [(chartAt H _).isOpen_image_iff_of_subset_source inter_subset_left]
  /-
    🎉 no goals
  -/


/-- `achart H x` is the chart at `x`, considered as an element of the atlas.
Especially useful for working with `BasicSmoothVectorBundleCore`. -/
def achart (x : M) : atlas H M :=
  ⟨chartAt H x, chart_mem_atlas H x⟩


theorem achart_def (x : M) : achart H x = ⟨chartAt H x, chart_mem_atlas H x⟩ :=
  rfl


@[simp, mfld_simps]
theorem coe_achart (x : M) : (achart H x : PartialHomeomorph M H) = chartAt H x :=
  rfl


@[simp, mfld_simps]
theorem achart_val (x : M) : (achart H x).1 = chartAt H x :=
  rfl


theorem mem_achart_source (x : M) : x ∈ (achart H x).1.source :=
  mem_chart_source H x


theorem ChartedSpace.secondCountable_of_countable_cover [SecondCountableTopology H] {s : Set M}
    (hs : ⋃ (x) (_ : x ∈ s), (chartAt H x).source = univ) (hsc : s.Countable) :
    SecondCountableTopology M := by
  haveI : ∀ x : M, SecondCountableTopology (chartAt H x).source :=
    fun x ↦ (chartAt (H := H) x).secondCountableTopology_source
  /-
    H : Type u
    M : Type u_2
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SecondCountableTopology H
    s : Set M
    hs : Eq (Set.iUnion fun x => Set.iUnion fun x_1 => (chartAt H x).source) Set.u …
    hsc : s.Countable
    this : ∀ (x : M), SecondCountableTopology ↑(chartAt H x).source
    ⊢ SecondCountableTopology M
  -/
  haveI := hsc.toEncodable
  /-
    H : Type u
    M : Type u_2
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SecondCountableTopology H
    s : Set M
    hs : Eq (Set.iUnion fun x => Set.iUnion fun x_1 => (chartAt H x).source) Set.u …
    hsc : s.Countable
    this✝ : ∀ (x : M), SecondCountableTopology ↑(chartAt H x).source
    this : Encodable ↑s
    ⊢ SecondCountableTopology M
  -/
  rw [biUnion_eq_iUnion] at hs
  /-
    H : Type u
    M : Type u_2
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SecondCountableTopology H
    s : Set M
    hs : Eq (Set.iUnion fun x => (chartAt H ↑x).source) Set.univ
    hsc : s.Countable
    this✝ : ∀ (x : M), SecondCountableTopology ↑(chartAt H x).source
    this : Encodable ↑s
    ⊢ SecondCountableTopology M
  -/
  exact secondCountableTopology_of_countable_cover (fun x : s ↦ (chartAt H (x : M)).open_source) hs
  /-
    🎉 no goals
  -/


theorem ChartedSpace.secondCountable_of_sigmaCompact [SecondCountableTopology H]
    [SigmaCompactSpace M] : SecondCountableTopology M := by
  obtain ⟨s, hsc, hsU⟩ : ∃ s, Set.Countable s ∧ ⋃ (x) (_ : x ∈ s), (chartAt H x).source = univ :=
    countable_cover_nhds_of_sigmaCompact fun x : M ↦ chart_source_mem_nhds H x
  /-
    case intro.intro
    H : Type u
    M : Type u_2
    inst✝⁴ : TopologicalSpace H
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SecondCountableTopology H
    inst✝ : SigmaCompactSpace M
    s : Set M
    hsc : s.Countable
    hsU : Eq (Set.iUnion fun x => Set.iUnion fun x_1 => (chartAt H x).source) Set. …
    ⊢ SecondCountableTopology M
  -/
  exact ChartedSpace.secondCountable_of_countable_cover H hsU hsc
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-13")] alias
ChartedSpace.secondCountable_of_sigma_compact := ChartedSpace.secondCountable_of_sigmaCompact


/-- If a topological space admits an atlas with locally compact charts, then the space itself
is locally compact. -/
theorem ChartedSpace.locallyCompactSpace [LocallyCompactSpace H] : LocallyCompactSpace M := by
  have : ∀ x : M, (𝓝 x).HasBasis
      (fun s ↦ s ∈ 𝓝 (chartAt H x x) ∧ IsCompact s ∧ s ⊆ (chartAt H x).target)
      fun s ↦ (chartAt H x).symm '' s := fun x ↦ by
    rw [← (chartAt H x).symm_map_nhds_eq (mem_chart_source H x)]
    exact ((compact_basis_nhds (chartAt H x x)).hasBasis_self_subset
      (chart_target_mem_nhds H x)).map _
  /-
    H : Type u
    M : Type u_2
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : LocallyCompactSpace H
    this : ∀ (x : M), (nhds x).HasBasis (fun s => And (Membership.mem (nhds (↑(cha …
    ⊢ LocallyCompactSpace M
  -/
  refine .of_hasBasis this ?_
  /-
    H : Type u
    M : Type u_2
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : LocallyCompactSpace H
    this : ∀ (x : M), (nhds x).HasBasis (fun s => And (Membership.mem (nhds (↑(cha …
    ⊢ ∀ (x : M) (i : Set H), And (Membership.mem (nhds (↑(chartAt H x) x)) i) (And …
  -/
  rintro x s ⟨_, h₂, h₃⟩
  /-
    case intro.intro
    H : Type u
    M : Type u_2
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : LocallyCompactSpace H
    this : ∀ (x : M), (nhds x).HasBasis (fun s => And (Membership.mem (nhds (↑(cha …
    x : M
    s : Set H
    left✝ : Membership.mem (nhds (↑(chartAt H x) x)) s
    h₂ : IsCompact s
    h₃ : HasSubset.Subset s (chartAt H x).target
    ⊢ IsCompact (Set.image (↑(chartAt H x).symm) s)
  -/
  exact h₂.image_of_continuousOn ((chartAt H x).continuousOn_symm.mono h₃)
  /-
    🎉 no goals
  -/


/-- If a topological space admits an atlas with locally connected charts, then the space itself is
locally connected. -/
theorem ChartedSpace.locallyConnectedSpace [LocallyConnectedSpace H] : LocallyConnectedSpace M := by
  /-
    H : Type u
    M : Type u_2
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : LocallyConnectedSpace H
    ⊢ LocallyConnectedSpace M
  -/
  let e : M → PartialHomeomorph M H := chartAt H
  refine locallyConnectedSpace_of_connected_bases (fun x s ↦ (e x).symm '' s)
      (fun x s ↦ (IsOpen s ∧ e x x ∈ s ∧ IsConnected s) ∧ s ⊆ (e x).target) ?_ ?_
    /-
      case refine_1
      H : Type u
      M : Type u_2
      inst✝³ : TopologicalSpace H
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      inst✝ : LocallyConnectedSpace H
      e : M → PartialHomeomorph M H := chartAt H
      ⊢ ∀ (x : M), (nhds x).HasBasis ((fun x s => And (And (IsOpen s) (And (Membersh …
    -/
  · intro x
    simpa only [e, PartialHomeomorph.symm_map_nhds_eq, mem_chart_source] using
      ((LocallyConnectedSpace.open_connected_basis (e x x)).restrict_subset
        ((e x).open_target.mem_nhds (mem_chart_target H x))).map (e x).symm
    /-
      case refine_2
      H : Type u
      M : Type u_2
      inst✝³ : TopologicalSpace H
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      inst✝ : LocallyConnectedSpace H
      e : M → PartialHomeomorph M H := chartAt H
      ⊢ ∀ (x : M) (i : Set H), (fun x s => And (And (IsOpen s) (And (Membership.mem  …
    -/
  · rintro x s ⟨⟨-, -, hsconn⟩, hssubset⟩
    /-
      case refine_2.intro.intro.intro
      H : Type u
      M : Type u_2
      inst✝³ : TopologicalSpace H
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      inst✝ : LocallyConnectedSpace H
      e : M → PartialHomeomorph M H := chartAt H
      x : M
      s : Set H
      hssubset : HasSubset.Subset s (e x).target
      hsconn : IsConnected s
      ⊢ IsPreconnected ((fun x s => Set.image (↑(e x).symm) s) x s)
    -/
    exact hsconn.isPreconnected.image _ ((e x).continuousOn_symm.mono hssubset)
    /-
      🎉 no goals
    -/


/-- If a topological space `M` admits an atlas with locally path-connected charts,
  then `M` itself is locally path-connected. -/
theorem ChartedSpace.locPathConnectedSpace [LocPathConnectedSpace H] : LocPathConnectedSpace M := by
  /-
    H : Type u
    M : Type u_2
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : LocPathConnectedSpace H
    ⊢ LocPathConnectedSpace M
  -/
  refine ⟨fun x ↦ ⟨fun s ↦ ⟨fun hs ↦ ?_, fun ⟨u, hu⟩ ↦ Filter.mem_of_superset hu.1.1 hu.2⟩⟩⟩
  /-
    H : Type u
    M : Type u_2
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : LocPathConnectedSpace H
    x : M
    s : Set M
    hs : Membership.mem (nhds x) s
    ⊢ Exists fun i => And (And (Membership.mem (nhds x) i) (IsPathConnected i)) (H …
  -/
  let e := chartAt H x
  /-
    H : Type u
    M : Type u_2
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : LocPathConnectedSpace H
    x : M
    s : Set M
    hs : Membership.mem (nhds x) s
    e : PartialHomeomorph M H := chartAt H x
    ⊢ Exists fun i => And (And (Membership.mem (nhds x) i) (IsPathConnected i)) (H …
  -/
  let t := s ∩ e.source
  /-
    H : Type u
    M : Type u_2
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : LocPathConnectedSpace H
    x : M
    s : Set M
    hs : Membership.mem (nhds x) s
    e : PartialHomeomorph M H := chartAt H x
    t : Set M := Inter.inter s e.source
    ⊢ Exists fun i => And (And (Membership.mem (nhds x) i) (IsPathConnected i)) (H …
  -/
  have ht : t ∈ 𝓝 x := Filter.inter_mem hs (chart_source_mem_nhds _ _)
  /-
    H : Type u
    M : Type u_2
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : LocPathConnectedSpace H
    x : M
    s : Set M
    hs : Membership.mem (nhds x) s
    e : PartialHomeomorph M H := chartAt H x
    t : Set M := Inter.inter s e.source
    ht : Membership.mem (nhds x) t
    ⊢ Exists fun i => And (And (Membership.mem (nhds x) i) (IsPathConnected i)) (H …
  -/
  refine ⟨e.symm '' pathComponentIn (e x) (e '' t), ⟨?_, ?_⟩, (?_ : _ ⊆ t).trans inter_subset_left⟩
    /-
      case refine_1
      H : Type u
      M : Type u_2
      inst✝³ : TopologicalSpace H
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      inst✝ : LocPathConnectedSpace H
      x : M
      s : Set M
      hs : Membership.mem (nhds x) s
      e : PartialHomeomorph M H := chartAt H x
      t : Set M := Inter.inter s e.source
      ht : Membership.mem (nhds x) t
      ⊢ Membership.mem (nhds x) (Set.image (↑e.symm) (pathComponentIn (↑e x) (Set.im …
    -/
  · nth_rewrite 1 [← e.left_inv (mem_chart_source _ _)]
    /-
      case refine_1
      H : Type u
      M : Type u_2
      inst✝³ : TopologicalSpace H
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      inst✝ : LocPathConnectedSpace H
      x : M
      s : Set M
      hs : Membership.mem (nhds x) s
      e : PartialHomeomorph M H := chartAt H x
      t : Set M := Inter.inter s e.source
      ht : Membership.mem (nhds x) t
      ⊢ Membership.mem (nhds (↑e.symm (↑e x))) (Set.image (↑e.symm) (pathComponentIn …
    -/
    apply e.symm.image_mem_nhds (by simp [e])
    /-
      case refine_1
      H : Type u
      M : Type u_2
      inst✝³ : TopologicalSpace H
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      inst✝ : LocPathConnectedSpace H
      x : M
      s : Set M
      hs : Membership.mem (nhds x) s
      e : PartialHomeomorph M H := chartAt H x
      t : Set M := Inter.inter s e.source
      ht : Membership.mem (nhds x) t
      ⊢ Membership.mem (nhds (↑e x)) (pathComponentIn (↑e x) (Set.image (↑e) t))
    -/
    exact pathComponentIn_mem_nhds <| e.image_mem_nhds (mem_chart_source _ _) ht
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      H : Type u
      M : Type u_2
      inst✝³ : TopologicalSpace H
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      inst✝ : LocPathConnectedSpace H
      x : M
      s : Set M
      hs : Membership.mem (nhds x) s
      e : PartialHomeomorph M H := chartAt H x
      t : Set M := Inter.inter s e.source
      ht : Membership.mem (nhds x) t
      ⊢ IsPathConnected (Set.image (↑e.symm) (pathComponentIn (↑e x) (Set.image (↑e) …
    -/
  · refine (isPathConnected_pathComponentIn <| mem_image_of_mem e (mem_of_mem_nhds ht)).image' ?_
    /-
      case refine_2
      H : Type u
      M : Type u_2
      inst✝³ : TopologicalSpace H
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      inst✝ : LocPathConnectedSpace H
      x : M
      s : Set M
      hs : Membership.mem (nhds x) s
      e : PartialHomeomorph M H := chartAt H x
      t : Set M := Inter.inter s e.source
      ht : Membership.mem (nhds x) t
      ⊢ ContinuousOn (↑e.symm) (pathComponentIn (↑e x) (Set.image (↑e) t))
    -/
    refine e.continuousOn_symm.mono <| subset_trans ?_ e.map_source''
    /-
      case refine_2
      H : Type u
      M : Type u_2
      inst✝³ : TopologicalSpace H
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      inst✝ : LocPathConnectedSpace H
      x : M
      s : Set M
      hs : Membership.mem (nhds x) s
      e : PartialHomeomorph M H := chartAt H x
      t : Set M := Inter.inter s e.source
      ht : Membership.mem (nhds x) t
      ⊢ HasSubset.Subset (pathComponentIn (↑e x) (Set.image (↑e) t)) (Set.image (↑e) …
    -/
    exact (pathComponentIn_mono <| image_mono inter_subset_right).trans pathComponentIn_subset
    /-
      🎉 no goals
    -/
  · exact (image_mono pathComponentIn_subset).trans
      (PartialEquiv.symm_image_image_of_subset_source _ inter_subset_right).subset


/-- If `M` is modelled on `H'` and `H'` is itself modelled on `H`, then we can consider `M` as being
modelled on `H`. -/
def ChartedSpace.comp (H : Type*) [TopologicalSpace H] (H' : Type*) [TopologicalSpace H']
    (M : Type*) [TopologicalSpace M] [ChartedSpace H H'] [ChartedSpace H' M] :
    ChartedSpace H M where
  atlas := image2 PartialHomeomorph.trans (atlas H' M) (atlas H H')
  chartAt p := (chartAt H' p).trans (chartAt H (chartAt H' p p))
                           /-
                             H✝ : Type u
                             H'✝ : Type u_1
                             M✝ : Type u_2
                             M' : Type u_3
                             M'' : Type u_4
                             inst✝⁷ : TopologicalSpace H✝
                             inst✝⁶ : TopologicalSpace M✝
                             inst✝⁵ : ChartedSpace H✝ M✝
                             H : Type u_5
                             inst✝⁴ : TopologicalSpace H
                             H' : Type u_6
                             inst✝³ : TopologicalSpace H'
                             M : Type u_7
                             inst✝² : TopologicalSpace M
                             inst✝¹ : ChartedSpace H H'
                             inst✝ : ChartedSpace H' M
                             p : M
                             ⊢ Membership.mem ((fun p => (chartAt H' p).trans (chartAt H (↑(chartAt H' p) p …
                           -/
  mem_chart_source p := by simp only [mfld_simps]
                           /-
                             🎉 no goals
                           -/
  chart_mem_atlas p := ⟨chartAt _ p, chart_mem_atlas _ p, chartAt _ _, chart_mem_atlas _ _, rfl⟩


theorem chartAt_comp (H : Type*) [TopologicalSpace H] (H' : Type*) [TopologicalSpace H']
    {M : Type*} [TopologicalSpace M] [ChartedSpace H H'] [ChartedSpace H' M] (x : M) :
    (letI := ChartedSpace.comp H H' M; chartAt H x) = chartAt H' x ≫ₕ chartAt H (chartAt H' x x) :=
  rfl


/-- A charted space over a T1 space is T1. Note that this is *not* true for T2 (for instance for
the real line with a double origin). -/
theorem ChartedSpace.t1Space [T1Space H] : T1Space M := by
  /-
    H : Type u
    M : Type u_2
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : T1Space H
    ⊢ T1Space M
  -/
  apply t1Space_iff_exists_open.2 (fun x y hxy ↦ ?_)
  /-
    H : Type u
    M : Type u_2
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : T1Space H
    x y : M
    hxy : Ne x y
    ⊢ Exists fun U => And (IsOpen U) (And (Membership.mem U x) (Not (Membership.me …
  -/
  by_cases hy : y ∈ (chartAt H x).source
    /-
      case pos
      H : Type u
      M : Type u_2
      inst✝³ : TopologicalSpace H
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      inst✝ : T1Space H
      x y : M
      hxy : Ne x y
      hy : Membership.mem (chartAt H x).source y
      ⊢ Exists fun U => And (IsOpen U) (And (Membership.mem U x) (Not (Membership.me …
    -/
  · refine ⟨(chartAt H x).source ∩ (chartAt H x)⁻¹' ({chartAt H x y}ᶜ), ?_, ?_, by simp⟩
      /-
        case pos.refine_1
        H : Type u
        M : Type u_2
        inst✝³ : TopologicalSpace H
        inst✝² : TopologicalSpace M
        inst✝¹ : ChartedSpace H M
        inst✝ : T1Space H
        x y : M
        hxy : Ne x y
        hy : Membership.mem (chartAt H x).source y
        ⊢ IsOpen (Inter.inter (chartAt H x).source (Set.preimage (↑(chartAt H x)) (Has …
      -/
    · exact PartialHomeomorph.isOpen_inter_preimage _ isOpen_compl_singleton
      /-
        🎉 no goals
      -/
    · simp only [preimage_compl, mem_inter_iff, mem_chart_source, mem_compl_iff, mem_preimage,
        mem_singleton_iff, true_and]
      /-
        case pos.refine_2
        H : Type u
        M : Type u_2
        inst✝³ : TopologicalSpace H
        inst✝² : TopologicalSpace M
        inst✝¹ : ChartedSpace H M
        inst✝ : T1Space H
        x y : M
        hxy : Ne x y
        hy : Membership.mem (chartAt H x).source y
        ⊢ Not (Eq (↑(chartAt H x) x) (↑(chartAt H x) y))
      -/
      exact (chartAt H x).injOn.ne (ChartedSpace.mem_chart_source x) hy hxy
      /-
        🎉 no goals
      -/
    /-
      case neg
      H : Type u
      M : Type u_2
      inst✝³ : TopologicalSpace H
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      inst✝ : T1Space H
      x y : M
      hxy : Ne x y
      hy : Not (Membership.mem (chartAt H x).source y)
      ⊢ Exists fun U => And (IsOpen U) (And (Membership.mem U x) (Not (Membership.me …
    -/
  · exact ⟨(chartAt H x).source, (chartAt H x).open_source, ChartedSpace.mem_chart_source x, hy⟩
    /-
      🎉 no goals
    -/


/-- Same thing as `H × H'`. We introduce it for technical reasons,
see note [Manifold type tags]. -/
def ModelProd (H : Type*) (H' : Type*) :=
  H × H'


/-- Same thing as `∀ i, H i`. We introduce it for technical reasons,
see note [Manifold type tags]. -/
def ModelPi {ι : Type*} (H : ι → Type*) :=
  ∀ i, H i


instance modelProdInhabited [Inhabited H] [Inhabited H'] : Inhabited (ModelProd H H') :=
  instInhabitedProd


instance (H : Type*) [TopologicalSpace H] (H' : Type*) [TopologicalSpace H'] :
    TopologicalSpace (ModelProd H H') :=
  instTopologicalSpaceProd

-- Porting note: simpNF false positive
-- Next lemma shows up often when dealing with derivatives, register it as simp.

@[simp, mfld_simps, nolint simpNF]
theorem modelProd_range_prod_id {H : Type*} {H' : Type*} {α : Type*} (f : H → α) :
    (range fun p : ModelProd H H' ↦ (f p.1, p.2)) = range f ×ˢ (univ : Set H') := by
  /-
    H : Type u_5
    H' : Type u_6
    α : Type u_7
    f : H → α
    ⊢ Eq (Set.range fun p => { fst := f p.1, snd := p.2 }) (SProd.sprod (Set.range …
  -/
  rw [prod_range_univ_eq]
  /-
    H : Type u_5
    H' : Type u_6
    α : Type u_7
    f : H → α
    ⊢ Eq (Set.range fun p => { fst := f p.1, snd := p.2 }) (Set.range fun p => { f …
  -/
  rfl
  /-
    🎉 no goals
  -/


instance modelPiInhabited [∀ i, Inhabited (Hi i)] : Inhabited (ModelPi Hi) :=
  Pi.instInhabited


instance [∀ i, TopologicalSpace (Hi i)] : TopologicalSpace (ModelPi Hi) :=
  Pi.topologicalSpace


/-- The product of two charted spaces is naturally a charted space, with the canonical
construction of the atlas of product maps. -/
instance prodChartedSpace (H : Type*) [TopologicalSpace H] (M : Type*) [TopologicalSpace M]
    [ChartedSpace H M] (H' : Type*) [TopologicalSpace H'] (M' : Type*) [TopologicalSpace M']
    [ChartedSpace H' M'] : ChartedSpace (ModelProd H H') (M × M') where
  atlas := image2 PartialHomeomorph.prod (atlas H M) (atlas H' M')
  chartAt x := (chartAt H x.1).prod (chartAt H' x.2)
  mem_chart_source x := ⟨mem_chart_source H x.1, mem_chart_source H' x.2⟩
  chart_mem_atlas x := mem_image2_of_mem (chart_mem_atlas H x.1) (chart_mem_atlas H' x.2)


@[ext]
theorem ModelProd.ext {x y : ModelProd H H'} (h₁ : x.1 = y.1) (h₂ : x.2 = y.2) : x = y :=
  Prod.ext h₁ h₂


@[simp, mfld_simps]
theorem prodChartedSpace_chartAt :
    chartAt (ModelProd H H') x = (chartAt H x.fst).prod (chartAt H' x.snd) :=
  rfl


theorem chartedSpaceSelf_prod : prodChartedSpace H H H' H' = chartedSpaceSelf (H × H') := by
  /-
    H : Type u
    H' : Type u_1
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace H'
    ⊢ Eq (prodChartedSpace H H H' H') (chartedSpaceSelf (Prod H H'))
  -/
  ext1
    /-
      case atlas
      H : Type u
      H' : Type u_1
      inst✝¹ : TopologicalSpace H
      inst✝ : TopologicalSpace H'
      ⊢ Eq ChartedSpace.atlas ChartedSpace.atlas
    -/
  · simp [prodChartedSpace, atlas, ChartedSpace.atlas]
    /-
      🎉 no goals
    -/
    /-
      case chartAt
      H : Type u
      H' : Type u_1
      inst✝¹ : TopologicalSpace H
      inst✝ : TopologicalSpace H'
      ⊢ Eq ChartedSpace.chartAt ChartedSpace.chartAt
    -/
  · ext1
    /-
      case chartAt.h
      H : Type u
      H' : Type u_1
      inst✝¹ : TopologicalSpace H
      inst✝ : TopologicalSpace H'
      x✝ : Prod H H'
      ⊢ Eq (ChartedSpace.chartAt x✝) (ChartedSpace.chartAt x✝)
    -/
    simp only [prodChartedSpace_chartAt, chartAt_self_eq, refl_prod_refl]
    /-
      case chartAt.h
      H : Type u
      H' : Type u_1
      inst✝¹ : TopologicalSpace H
      inst✝ : TopologicalSpace H'
      x✝ : Prod H H'
      ⊢ Eq (PartialHomeomorph.refl (Prod H H')) (PartialHomeomorph.refl (ModelProd H …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- The product of a finite family of charted spaces is naturally a charted space, with the
canonical construction of the atlas of finite product maps. -/
instance piChartedSpace {ι : Type*} [Finite ι] (H : ι → Type*) [∀ i, TopologicalSpace (H i)]
    (M : ι → Type*) [∀ i, TopologicalSpace (M i)] [∀ i, ChartedSpace (H i) (M i)] :
    ChartedSpace (ModelPi H) (∀ i, M i) where
  atlas := PartialHomeomorph.pi '' Set.pi univ fun _ ↦ atlas (H _) (M _)
  chartAt f := PartialHomeomorph.pi fun i ↦ chartAt (H i) (f i)
  mem_chart_source f i _ := mem_chart_source (H i) (f i)
  chart_mem_atlas f := mem_image_of_mem _ fun i _ ↦ chart_mem_atlas (H i) (f i)


@[simp, mfld_simps]
theorem piChartedSpace_chartAt {ι : Type*} [Finite ι] (H : ι → Type*)
    [∀ i, TopologicalSpace (H i)] (M : ι → Type*) [∀ i, TopologicalSpace (M i)]
    [∀ i, ChartedSpace (H i) (M i)] (f : ∀ i, M i) :
    chartAt (H := ModelPi H) f = PartialHomeomorph.pi fun i ↦ chartAt (H i) (f i) :=
  rfl


/-- Sometimes, one may want to construct a charted space structure on a space which does not yet
have a topological structure, where the topology would come from the charts. For this, one needs
charts that are only partial equivalences, and continuity properties for their composition.
This is formalised in `ChartedSpaceCore`. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): this linter isn't ported yet.
-- @[nolint has_nonempty_instance]
structure ChartedSpaceCore (H : Type*) [TopologicalSpace H] (M : Type*) where
  /-- An atlas of charts, which are only `PartialEquiv`s -/
  atlas : Set (PartialEquiv M H)
  /-- The preferred chart at each point -/
  chartAt : M → PartialEquiv M H
  mem_chart_source : ∀ x, x ∈ (chartAt x).source
  chart_mem_atlas : ∀ x, chartAt x ∈ atlas
  open_source : ∀ e e' : PartialEquiv M H, e ∈ atlas → e' ∈ atlas → IsOpen (e.symm.trans e').source
  continuousOn_toFun : ∀ e e' : PartialEquiv M H, e ∈ atlas → e' ∈ atlas →
    ContinuousOn (e.symm.trans e') (e.symm.trans e').source


/-- Topology generated by a set of charts on a Type. -/
protected def toTopologicalSpace : TopologicalSpace M :=
  TopologicalSpace.generateFrom <|
    ⋃ (e : PartialEquiv M H) (_ : e ∈ c.atlas) (s : Set H) (_ : IsOpen s),
      {e ⁻¹' s ∩ e.source}


theorem open_source' (he : e ∈ c.atlas) : IsOpen[c.toTopologicalSpace] e.source := by
  /-
    H : Type u
    M : Type u_2
    inst✝ : TopologicalSpace H
    c : ChartedSpaceCore H M
    e : PartialEquiv M H
    he : Membership.mem c.atlas e
    ⊢ IsOpen e.source
  -/
  apply TopologicalSpace.GenerateOpen.basic
  /-
    case a
    H : Type u
    M : Type u_2
    inst✝ : TopologicalSpace H
    c : ChartedSpaceCore H M
    e : PartialEquiv M H
    he : Membership.mem c.atlas e
    ⊢ Membership.mem (Set.iUnion fun e => Set.iUnion fun x => Set.iUnion fun s =>  …
  -/
  simp only [exists_prop, mem_iUnion, mem_singleton_iff]
  /-
    case a
    H : Type u
    M : Type u_2
    inst✝ : TopologicalSpace H
    c : ChartedSpaceCore H M
    e : PartialEquiv M H
    he : Membership.mem c.atlas e
    ⊢ Exists fun i => And (Membership.mem c.atlas i) (Exists fun i_1 => And (IsOpe …
  -/
  refine ⟨e, he, univ, isOpen_univ, ?_⟩
  /-
    case a
    H : Type u
    M : Type u_2
    inst✝ : TopologicalSpace H
    c : ChartedSpaceCore H M
    e : PartialEquiv M H
    he : Membership.mem c.atlas e
    ⊢ Eq e.source (Inter.inter (Set.preimage (↑e) Set.univ) e.source)
  -/
  simp only [Set.univ_inter, Set.preimage_univ]
  /-
    🎉 no goals
  -/


theorem open_target (he : e ∈ c.atlas) : IsOpen e.target := by
  have E : e.target ∩ e.symm ⁻¹' e.source = e.target :=
    Subset.antisymm inter_subset_left fun x hx ↦
      ⟨hx, PartialEquiv.target_subset_preimage_source _ hx⟩
  /-
    H : Type u
    M : Type u_2
    inst✝ : TopologicalSpace H
    c : ChartedSpaceCore H M
    e : PartialEquiv M H
    he : Membership.mem c.atlas e
    E : Eq (Inter.inter e.target (Set.preimage (↑e.symm) e.source)) e.target
    ⊢ IsOpen e.target
  -/
  simpa [PartialEquiv.trans_source, E] using c.open_source e e he he
  /-
    🎉 no goals
  -/


/-- An element of the atlas in a charted space without topology becomes a partial homeomorphism
for the topology constructed from this atlas. The `PartialHomeomorph` version is given in this
definition. -/
protected def partialHomeomorph (e : PartialEquiv M H) (he : e ∈ c.atlas) :
    @PartialHomeomorph M H c.toTopologicalSpace _ :=
  { __ := c.toTopologicalSpace
    __ := e
                      /-
                        H : Type u
                        H' : Type u_1
                        M : Type u_2
                        M' : Type u_3
                        M'' : Type u_4
                        inst✝ : TopologicalSpace H
                        c : ChartedSpaceCore H M
                        e✝ e : PartialEquiv M H
                        he : Membership.mem c.atlas e
                        ⊢ IsOpen __spread✝¹⁻⁰.source
                      -/
    open_source := by convert c.open_source' he
                      /-
                        🎉 no goals
                      -/
                      /-
                        H : Type u
                        H' : Type u_1
                        M : Type u_2
                        M' : Type u_3
                        M'' : Type u_4
                        inst✝ : TopologicalSpace H
                        c : ChartedSpaceCore H M
                        e✝ e : PartialEquiv M H
                        he : Membership.mem c.atlas e
                        ⊢ IsOpen __spread✝¹⁻⁰.target
                      -/
    open_target := by convert c.open_target he
                      /-
                        🎉 no goals
                      -/
    continuousOn_toFun := by
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        c : ChartedSpaceCore H M
        e✝ e : PartialEquiv M H
        he : Membership.mem c.atlas e
        ⊢ ContinuousOn (↑__spread✝¹⁻⁰) __spread✝¹⁻⁰.source
      -/
      letI : TopologicalSpace M := c.toTopologicalSpace
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        c : ChartedSpaceCore H M
        e✝ e : PartialEquiv M H
        he : Membership.mem c.atlas e
        this : TopologicalSpace M := c.toTopologicalSpace
        ⊢ ContinuousOn (↑__spread✝¹⁻⁰) __spread✝¹⁻⁰.source
      -/
      rw [continuousOn_open_iff (c.open_source' he)]
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        c : ChartedSpaceCore H M
        e✝ e : PartialEquiv M H
        he : Membership.mem c.atlas e
        this : TopologicalSpace M := c.toTopologicalSpace
        ⊢ ∀ (t : Set H), IsOpen t → IsOpen (Inter.inter e.source (Set.preimage (↑__spr …
      -/
      intro s s_open
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        c : ChartedSpaceCore H M
        e✝ e : PartialEquiv M H
        he : Membership.mem c.atlas e
        this : TopologicalSpace M := c.toTopologicalSpace
        s : Set H
        s_open : IsOpen s
        ⊢ IsOpen (Inter.inter e.source (Set.preimage (↑__spread✝¹⁻⁰) s))
      -/
      rw [inter_comm]
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        c : ChartedSpaceCore H M
        e✝ e : PartialEquiv M H
        he : Membership.mem c.atlas e
        this : TopologicalSpace M := c.toTopologicalSpace
        s : Set H
        s_open : IsOpen s
        ⊢ IsOpen (Inter.inter (Set.preimage (↑__spread✝¹⁻⁰) s) e.source)
      -/
      apply TopologicalSpace.GenerateOpen.basic
      /-
        case a
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        c : ChartedSpaceCore H M
        e✝ e : PartialEquiv M H
        he : Membership.mem c.atlas e
        this : TopologicalSpace M := c.toTopologicalSpace
        s : Set H
        s_open : IsOpen s
        ⊢ Membership.mem (Set.iUnion fun e => Set.iUnion fun x => Set.iUnion fun s =>  …
      -/
      simp only [exists_prop, mem_iUnion, mem_singleton_iff]
      /-
        case a
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        c : ChartedSpaceCore H M
        e✝ e : PartialEquiv M H
        he : Membership.mem c.atlas e
        this : TopologicalSpace M := c.toTopologicalSpace
        s : Set H
        s_open : IsOpen s
        ⊢ Exists fun i => And (Membership.mem c.atlas i) (Exists fun i_1 => And (IsOpe …
      -/
      exact ⟨e, he, ⟨s, s_open, rfl⟩⟩
      /-
        🎉 no goals
      -/
    continuousOn_invFun := by
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        c : ChartedSpaceCore H M
        e✝ e : PartialEquiv M H
        he : Membership.mem c.atlas e
        ⊢ ContinuousOn __spread✝¹⁻⁰.invFun __spread✝¹⁻⁰.target
      -/
      letI : TopologicalSpace M := c.toTopologicalSpace
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        c : ChartedSpaceCore H M
        e✝ e : PartialEquiv M H
        he : Membership.mem c.atlas e
        this : TopologicalSpace M := c.toTopologicalSpace
        ⊢ ContinuousOn __spread✝¹⁻⁰.invFun __spread✝¹⁻⁰.target
      -/
      apply continuousOn_isOpen_of_generateFrom
      /-
        case h
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        c : ChartedSpaceCore H M
        e✝ e : PartialEquiv M H
        he : Membership.mem c.atlas e
        this : TopologicalSpace M := c.toTopologicalSpace
        ⊢ ∀ (t : Set M), Membership.mem (Set.iUnion fun e => Set.iUnion fun x => Set.i …
      -/
      intro t ht
      /-
        case h
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        c : ChartedSpaceCore H M
        e✝ e : PartialEquiv M H
        he : Membership.mem c.atlas e
        this : TopologicalSpace M := c.toTopologicalSpace
        t : Set M
        ht : Membership.mem (Set.iUnion fun e => Set.iUnion fun x => Set.iUnion fun s  …
        ⊢ IsOpen (Inter.inter __spread✝¹⁻⁰.target (Set.preimage __spread✝¹⁻⁰.invFun t))
      -/
      simp only [exists_prop, mem_iUnion, mem_singleton_iff] at ht
      /-
        case h
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        c : ChartedSpaceCore H M
        e✝ e : PartialEquiv M H
        he : Membership.mem c.atlas e
        this : TopologicalSpace M := c.toTopologicalSpace
        t : Set M
        ht : Exists fun i => And (Membership.mem c.atlas i) (Exists fun i_1 => And (Is …
        ⊢ IsOpen (Inter.inter __spread✝¹⁻⁰.target (Set.preimage __spread✝¹⁻⁰.invFun t))
      -/
      rcases ht with ⟨e', e'_atlas, s, s_open, ts⟩
      /-
        case h.intro.intro.intro.intro
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        c : ChartedSpaceCore H M
        e✝ e : PartialEquiv M H
        he : Membership.mem c.atlas e
        this : TopologicalSpace M := c.toTopologicalSpace
        t : Set M
        e' : PartialEquiv M H
        e'_atlas : Membership.mem c.atlas e'
        s : Set H
        s_open : IsOpen s
        ts : Eq t (Inter.inter (Set.preimage (↑e') s) e'.source)
        ⊢ IsOpen (Inter.inter __spread✝¹⁻⁰.target (Set.preimage __spread✝¹⁻⁰.invFun t))
      -/
      rw [ts]
      /-
        case h.intro.intro.intro.intro
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        c : ChartedSpaceCore H M
        e✝ e : PartialEquiv M H
        he : Membership.mem c.atlas e
        this : TopologicalSpace M := c.toTopologicalSpace
        t : Set M
        e' : PartialEquiv M H
        e'_atlas : Membership.mem c.atlas e'
        s : Set H
        s_open : IsOpen s
        ts : Eq t (Inter.inter (Set.preimage (↑e') s) e'.source)
        ⊢ IsOpen (Inter.inter __spread✝¹⁻⁰.target (Set.preimage __spread✝¹⁻⁰.invFun (I …
      -/
      let f := e.symm.trans e'
      have : IsOpen (f ⁻¹' s ∩ f.source) := by
        simpa [f, inter_comm] using (continuousOn_open_iff (c.open_source e e' he e'_atlas)).1
          (c.continuousOn_toFun e e' he e'_atlas) s s_open
      have A : e' ∘ e.symm ⁻¹' s ∩ (e.target ∩ e.symm ⁻¹' e'.source) =
          e.target ∩ (e' ∘ e.symm ⁻¹' s ∩ e.symm ⁻¹' e'.source) := by
        rw [← inter_assoc, ← inter_assoc]
        congr 1
        exact inter_comm _ _
      /-
        case h.intro.intro.intro.intro
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        c : ChartedSpaceCore H M
        e✝ e : PartialEquiv M H
        he : Membership.mem c.atlas e
        this✝ : TopologicalSpace M := c.toTopologicalSpace
        t : Set M
        e' : PartialEquiv M H
        e'_atlas : Membership.mem c.atlas e'
        s : Set H
        s_open : IsOpen s
        ts : Eq t (Inter.inter (Set.preimage (↑e') s) e'.source)
        f : PartialEquiv H H := e.symm.trans e'
        this : IsOpen (Inter.inter (Set.preimage (↑f) s) f.source)
        A : Eq (Inter.inter (Set.preimage (Function.comp ↑e' ↑e.symm) s) (Inter.inter  …
        ⊢ IsOpen (Inter.inter __spread✝¹⁻⁰.target (Set.preimage __spread✝¹⁻⁰.invFun (I …
      -/
      simpa [f, PartialEquiv.trans_source, preimage_inter, preimage_comp.symm, A] using this }
      /-
        🎉 no goals
      -/


/-- Given a charted space without topology, endow it with a genuine charted space structure with
respect to the topology constructed from the atlas. -/
def toChartedSpace : @ChartedSpace H _ M c.toTopologicalSpace :=
  { __ := c.toTopologicalSpace
    atlas := ⋃ (e : PartialEquiv M H) (he : e ∈ c.atlas), {c.partialHomeomorph e he}
    chartAt := fun x ↦ c.partialHomeomorph (c.chartAt x) (c.chart_mem_atlas x)
    mem_chart_source := fun x ↦ c.mem_chart_source x
    chart_mem_atlas := fun x ↦ by
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        c : ChartedSpaceCore H M
        e : PartialEquiv M H
        x : M
        ⊢ Membership.mem (Set.iUnion fun e => Set.iUnion fun he => Singleton.singleton …
      -/
      simp only [mem_iUnion, mem_singleton_iff]
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝ : TopologicalSpace H
        c : ChartedSpaceCore H M
        e : PartialEquiv M H
        x : M
        ⊢ Exists fun i => Exists fun h => Eq (c.partialHomeomorph (c.chartAt x) ⋯) (c. …
      -/
      exact ⟨c.chartAt x, c.chart_mem_atlas x, rfl⟩}
      /-
        🎉 no goals
      -/


/-- A charted space has an atlas in a groupoid `G` if the change of coordinates belong to the
groupoid. -/
class HasGroupoid {H : Type*} [TopologicalSpace H] (M : Type*) [TopologicalSpace M]
    [ChartedSpace H M] (G : StructureGroupoid H) : Prop where
  compatible : ∀ {e e' : PartialHomeomorph M H}, e ∈ atlas H M → e' ∈ atlas H M → e.symm ≫ₕ e' ∈ G


/-- Reformulate in the `StructureGroupoid` namespace the compatibility condition of charts in a
charted space admitting a structure groupoid, to make it more easily accessible with dot
notation. -/
theorem StructureGroupoid.compatible {H : Type*} [TopologicalSpace H] (G : StructureGroupoid H)
    {M : Type*} [TopologicalSpace M] [ChartedSpace H M] [HasGroupoid M G]
    {e e' : PartialHomeomorph M H} (he : e ∈ atlas H M) (he' : e' ∈ atlas H M) : e.symm ≫ₕ e' ∈ G :=
  HasGroupoid.compatible he he'


theorem hasGroupoid_of_le {G₁ G₂ : StructureGroupoid H} (h : HasGroupoid M G₁) (hle : G₁ ≤ G₂) :
    HasGroupoid M G₂ :=
  ⟨fun he he' ↦ hle (h.compatible he he')⟩


theorem hasGroupoid_inf_iff {G₁ G₂ : StructureGroupoid H} : HasGroupoid M (G₁ ⊓ G₂) ↔
    HasGroupoid M G₁ ∧ HasGroupoid M G₂ :=
  ⟨(fun h ↦ ⟨hasGroupoid_of_le h inf_le_left, hasGroupoid_of_le h inf_le_right⟩),
  fun ⟨h1, h2⟩ ↦ { compatible := fun he he' ↦ ⟨h1.compatible he he', h2.compatible he he'⟩ }⟩


theorem hasGroupoid_of_pregroupoid (PG : Pregroupoid H) (h : ∀ {e e' : PartialHomeomorph M H},
    e ∈ atlas H M → e' ∈ atlas H M → PG.property (e.symm ≫ₕ e') (e.symm ≫ₕ e').source) :
    HasGroupoid M PG.groupoid :=
  ⟨fun he he' ↦ mem_groupoid_of_pregroupoid.mpr ⟨h he he', h he' he⟩⟩


/-- The trivial charted space structure on the model space is compatible with any groupoid. -/
instance hasGroupoid_model_space (H : Type*) [TopologicalSpace H] (G : StructureGroupoid H) :
    HasGroupoid H G where
  compatible {e e'} he he' := by
    /-
      H✝ : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝³ : TopologicalSpace H✝
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H✝ M
      H : Type u_5
      inst✝ : TopologicalSpace H
      G : StructureGroupoid H
      e e' : PartialHomeomorph H H
      he : Membership.mem (atlas H H) e
      he' : Membership.mem (atlas H H) e'
      ⊢ Membership.mem G (e.symm.trans e')
    -/
    rw [chartedSpaceSelf_atlas] at he he'
    /-
      H✝ : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝³ : TopologicalSpace H✝
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H✝ M
      H : Type u_5
      inst✝ : TopologicalSpace H
      G : StructureGroupoid H
      e e' : PartialHomeomorph H H
      he : Eq e (PartialHomeomorph.refl H)
      he' : Eq e' (PartialHomeomorph.refl H)
      ⊢ Membership.mem G (e.symm.trans e')
    -/
    simp [he, he', StructureGroupoid.id_mem]
    /-
      🎉 no goals
    -/


/-- Any charted space structure is compatible with the groupoid of all partial homeomorphisms. -/
instance hasGroupoid_continuousGroupoid : HasGroupoid M (continuousGroupoid H) := by
  /-
    H : Type u
    H' : Type u_1
    M : Type u_2
    M' : Type u_3
    M'' : Type u_4
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    ⊢ HasGroupoid M (continuousGroupoid H)
  -/
  refine ⟨fun _ _ ↦ ?_⟩
  /-
    H : Type u
    H' : Type u_1
    M : Type u_2
    M' : Type u_3
    M'' : Type u_4
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    e✝ e'✝ : PartialHomeomorph M H
    x✝¹ : Membership.mem (atlas H M) e✝
    x✝ : Membership.mem (atlas H M) e'✝
    ⊢ Membership.mem (continuousGroupoid H) (e✝.symm.trans e'✝)
  -/
  rw [continuousGroupoid, mem_groupoid_of_pregroupoid]
  /-
    H : Type u
    H' : Type u_1
    M : Type u_2
    M' : Type u_3
    M'' : Type u_4
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    e✝ e'✝ : PartialHomeomorph M H
    x✝¹ : Membership.mem (atlas H M) e✝
    x✝ : Membership.mem (atlas H M) e'✝
    ⊢ And ((continuousPregroupoid H).property (↑(e✝.symm.trans e'✝)) (e✝.symm.tran …
  -/
  simp only [and_self_iff]
  /-
    🎉 no goals
  -/


/-- If `G` is closed under restriction, the transition function between
  the restriction of two charts `e` and `e'` lies in `G`. -/
theorem StructureGroupoid.trans_restricted {e e' : PartialHomeomorph M H} {G : StructureGroupoid H}
    (he : e ∈ atlas H M) (he' : e' ∈ atlas H M)
    [HasGroupoid M G] [ClosedUnderRestriction G] {s : Opens M} (hs : Nonempty s) :
    (e.subtypeRestr hs).symm ≫ₕ e'.subtypeRestr hs ∈ G :=
  G.mem_of_eqOnSource (closedUnderRestriction' (G.compatible he he')
    (e.isOpen_inter_preimage_symm s.2)) (e.subtypeRestr_symm_trans_subtypeRestr hs e')


variable (M) in
/-- Given a charted space admitting a structure groupoid, the maximal atlas associated to this
structure groupoid is the set of all charts that are compatible with the atlas, i.e., such
that changing coordinates with an atlas member gives an element of the groupoid. -/
def StructureGroupoid.maximalAtlas : Set (PartialHomeomorph M H) :=
  { e | ∀ e' ∈ atlas H M, e.symm ≫ₕ e' ∈ G ∧ e'.symm ≫ₕ e ∈ G }


/-- The elements of the atlas belong to the maximal atlas for any structure groupoid. -/
theorem StructureGroupoid.subset_maximalAtlas [HasGroupoid M G] : atlas H M ⊆ G.maximalAtlas M :=
  fun _ he _ he' ↦ ⟨G.compatible he he', G.compatible he' he⟩


theorem StructureGroupoid.chart_mem_maximalAtlas [HasGroupoid M G] (x : M) :
    chartAt H x ∈ G.maximalAtlas M :=
  G.subset_maximalAtlas (chart_mem_atlas H x)


theorem mem_maximalAtlas_iff {e : PartialHomeomorph M H} :
    e ∈ G.maximalAtlas M ↔ ∀ e' ∈ atlas H M, e.symm ≫ₕ e' ∈ G ∧ e'.symm ≫ₕ e ∈ G :=
  Iff.rfl


/-- Changing coordinates between two elements of the maximal atlas gives rise to an element
of the structure groupoid. -/
theorem StructureGroupoid.compatible_of_mem_maximalAtlas {e e' : PartialHomeomorph M H}
    (he : e ∈ G.maximalAtlas M) (he' : e' ∈ G.maximalAtlas M) : e.symm ≫ₕ e' ∈ G := by
  /-
    H : Type u
    M : Type u_2
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    G : StructureGroupoid H
    e e' : PartialHomeomorph M H
    he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
    he' : Membership.mem (StructureGroupoid.maximalAtlas M G) e'
    ⊢ Membership.mem G (e.symm.trans e')
  -/
  refine G.locality fun x hx ↦ ?_
  /-
    H : Type u
    M : Type u_2
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    G : StructureGroupoid H
    e e' : PartialHomeomorph M H
    he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
    he' : Membership.mem (StructureGroupoid.maximalAtlas M G) e'
    x : H
    hx : Membership.mem (e.symm.trans e').source x
    ⊢ Exists fun s => And (IsOpen s) (And (Membership.mem s x) (Membership.mem G ( …
  -/
  set f := chartAt (H := H) (e.symm x)
  /-
    H : Type u
    M : Type u_2
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    G : StructureGroupoid H
    e e' : PartialHomeomorph M H
    he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
    he' : Membership.mem (StructureGroupoid.maximalAtlas M G) e'
    x : H
    hx : Membership.mem (e.symm.trans e').source x
    f : PartialHomeomorph M H := chartAt H (↑e.symm x)
    ⊢ Exists fun s => And (IsOpen s) (And (Membership.mem s x) (Membership.mem G ( …
  -/
  let s := e.target ∩ e.symm ⁻¹' f.source
  have hs : IsOpen s := by
    apply e.symm.continuousOn_toFun.isOpen_inter_preimage <;> apply open_source
  have xs : x ∈ s := by
    simp only [s, f, mem_inter_iff, mem_preimage, mem_chart_source, and_true]
    exact ((mem_inter_iff _ _ _).1 hx).1
  /-
    H : Type u
    M : Type u_2
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    G : StructureGroupoid H
    e e' : PartialHomeomorph M H
    he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
    he' : Membership.mem (StructureGroupoid.maximalAtlas M G) e'
    x : H
    hx : Membership.mem (e.symm.trans e').source x
    f : PartialHomeomorph M H := chartAt H (↑e.symm x)
    s : Set H := Inter.inter e.target (Set.preimage (↑e.symm) f.source)
    hs : IsOpen s
    xs : Membership.mem s x
    ⊢ Exists fun s => And (IsOpen s) (And (Membership.mem s x) (Membership.mem G ( …
  -/
  refine ⟨s, hs, xs, ?_⟩
  /-
    H : Type u
    M : Type u_2
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    G : StructureGroupoid H
    e e' : PartialHomeomorph M H
    he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
    he' : Membership.mem (StructureGroupoid.maximalAtlas M G) e'
    x : H
    hx : Membership.mem (e.symm.trans e').source x
    f : PartialHomeomorph M H := chartAt H (↑e.symm x)
    s : Set H := Inter.inter e.target (Set.preimage (↑e.symm) f.source)
    hs : IsOpen s
    xs : Membership.mem s x
    ⊢ Membership.mem G ((e.symm.trans e').restr s)
  -/
  have A : e.symm ≫ₕ f ∈ G := (mem_maximalAtlas_iff.1 he f (chart_mem_atlas _ _)).1
  /-
    H : Type u
    M : Type u_2
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    G : StructureGroupoid H
    e e' : PartialHomeomorph M H
    he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
    he' : Membership.mem (StructureGroupoid.maximalAtlas M G) e'
    x : H
    hx : Membership.mem (e.symm.trans e').source x
    f : PartialHomeomorph M H := chartAt H (↑e.symm x)
    s : Set H := Inter.inter e.target (Set.preimage (↑e.symm) f.source)
    hs : IsOpen s
    xs : Membership.mem s x
    A : Membership.mem G (e.symm.trans f)
    ⊢ Membership.mem G ((e.symm.trans e').restr s)
  -/
  have B : f.symm ≫ₕ e' ∈ G := (mem_maximalAtlas_iff.1 he' f (chart_mem_atlas _ _)).2
  /-
    H : Type u
    M : Type u_2
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    G : StructureGroupoid H
    e e' : PartialHomeomorph M H
    he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
    he' : Membership.mem (StructureGroupoid.maximalAtlas M G) e'
    x : H
    hx : Membership.mem (e.symm.trans e').source x
    f : PartialHomeomorph M H := chartAt H (↑e.symm x)
    s : Set H := Inter.inter e.target (Set.preimage (↑e.symm) f.source)
    hs : IsOpen s
    xs : Membership.mem s x
    A : Membership.mem G (e.symm.trans f)
    B : Membership.mem G (f.symm.trans e')
    ⊢ Membership.mem G ((e.symm.trans e').restr s)
  -/
  have C : (e.symm ≫ₕ f) ≫ₕ f.symm ≫ₕ e' ∈ G := G.trans A B
  have D : (e.symm ≫ₕ f) ≫ₕ f.symm ≫ₕ e' ≈ (e.symm ≫ₕ e').restr s := calc
    (e.symm ≫ₕ f) ≫ₕ f.symm ≫ₕ e' = e.symm ≫ₕ (f ≫ₕ f.symm) ≫ₕ e' := by simp only [trans_assoc]
    _ ≈ e.symm ≫ₕ ofSet f.source f.open_source ≫ₕ e' :=
      EqOnSource.trans' (refl _) (EqOnSource.trans' (self_trans_symm _) (refl _))
    _ ≈ (e.symm ≫ₕ ofSet f.source f.open_source) ≫ₕ e' := by rw [trans_assoc]
    _ ≈ e.symm.restr s ≫ₕ e' := by rw [trans_of_set']; apply refl
    _ ≈ (e.symm ≫ₕ e').restr s := by rw [restr_trans]
  /-
    H : Type u
    M : Type u_2
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    G : StructureGroupoid H
    e e' : PartialHomeomorph M H
    he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
    he' : Membership.mem (StructureGroupoid.maximalAtlas M G) e'
    x : H
    hx : Membership.mem (e.symm.trans e').source x
    f : PartialHomeomorph M H := chartAt H (↑e.symm x)
    s : Set H := Inter.inter e.target (Set.preimage (↑e.symm) f.source)
    hs : IsOpen s
    xs : Membership.mem s x
    A : Membership.mem G (e.symm.trans f)
    B : Membership.mem G (f.symm.trans e')
    C : Membership.mem G ((e.symm.trans f).trans (f.symm.trans e'))
    D : HasEquiv.Equiv ((e.symm.trans f).trans (f.symm.trans e')) ((e.symm.trans e …
    ⊢ Membership.mem G ((e.symm.trans e').restr s)
  -/
  exact G.mem_of_eqOnSource C (Setoid.symm D)
  /-
    🎉 no goals
  -/


open PartialHomeomorph in
/-- The maximal atlas of a structure groupoid is stable under equivalence. -/
lemma StructureGroupoid.mem_maximalAtlas_of_eqOnSource {e e' : PartialHomeomorph M H} (h : e' ≈ e)
    (he : e ∈ G.maximalAtlas M) : e' ∈ G.maximalAtlas M := by
  /-
    H : Type u
    M : Type u_2
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    G : StructureGroupoid H
    e e' : PartialHomeomorph M H
    h : HasEquiv.Equiv e' e
    he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
    ⊢ Membership.mem (StructureGroupoid.maximalAtlas M G) e'
  -/
  intro e'' he''
  /-
    H : Type u
    M : Type u_2
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    G : StructureGroupoid H
    e e' : PartialHomeomorph M H
    h : HasEquiv.Equiv e' e
    he : Membership.mem (StructureGroupoid.maximalAtlas M G) e
    e'' : PartialHomeomorph M H
    he'' : Membership.mem (atlas H M) e''
    ⊢ And (Membership.mem G (e'.symm.trans e'')) (Membership.mem G (e''.symm.trans …
  -/
  obtain ⟨l, r⟩ := mem_maximalAtlas_iff.mp he e'' he''
  exact ⟨G.mem_of_eqOnSource l (EqOnSource.trans' (EqOnSource.symm' h) (e''.eqOnSource_refl)),
         G.mem_of_eqOnSource r (EqOnSource.trans' (e''.symm).eqOnSource_refl h)⟩


/-- In the model space, the identity is in any maximal atlas. -/
theorem StructureGroupoid.id_mem_maximalAtlas : PartialHomeomorph.refl H ∈ G.maximalAtlas H :=
                              /-
                                H : Type u
                                inst✝ : TopologicalSpace H
                                G : StructureGroupoid H
                                ⊢ Membership.mem (atlas H H) (PartialHomeomorph.refl H)
                              -/
  G.subset_maximalAtlas <| by simp
                              /-
                                🎉 no goals
                              -/


/-- In the model space, any element of the groupoid is in the maximal atlas. -/
theorem StructureGroupoid.mem_maximalAtlas_of_mem_groupoid {f : PartialHomeomorph H H}
    (hf : f ∈ G) : f ∈ G.maximalAtlas H := by
  /-
    H : Type u
    inst✝ : TopologicalSpace H
    G : StructureGroupoid H
    f : PartialHomeomorph H H
    hf : Membership.mem G f
    ⊢ Membership.mem (StructureGroupoid.maximalAtlas H G) f
  -/
  rintro e (rfl : e = PartialHomeomorph.refl H)
  /-
    H : Type u
    inst✝ : TopologicalSpace H
    G : StructureGroupoid H
    f : PartialHomeomorph H H
    hf : Membership.mem G f
    ⊢ And (Membership.mem G (f.symm.trans (PartialHomeomorph.refl H))) (Membership …
  -/
  exact ⟨G.trans (G.symm hf) G.id_mem, G.trans (G.symm G.id_mem) hf⟩
  /-
    🎉 no goals
  -/


theorem StructureGroupoid.maximalAtlas_mono {G G' : StructureGroupoid H} (h : G ≤ G') :
    G.maximalAtlas M ⊆ G'.maximalAtlas M :=
  fun _ he e' he' ↦ ⟨h (he e' he').1, h (he e' he').2⟩


/-- If a single partial homeomorphism `e` from a space `α` into `H` has source covering the whole
space `α`, then that partial homeomorphism induces an `H`-charted space structure on `α`.
(This condition is equivalent to `e` being an open embedding of `α` into `H`; see
`IsOpenEmbedding.singletonChartedSpace`.) -/
def singletonChartedSpace (h : e.source = Set.univ) : ChartedSpace H α where
  atlas := {e}
  chartAt _ := e
                           /-
                             H : Type u
                             H' : Type u_1
                             M : Type u_2
                             M' : Type u_3
                             M'' : Type u_4
                             inst✝³ : TopologicalSpace H
                             inst✝² : TopologicalSpace M
                             inst✝¹ : ChartedSpace H M
                             α : Type u_5
                             inst✝ : TopologicalSpace α
                             e : PartialHomeomorph α H
                             h : Eq e.source Set.univ
                             x✝ : α
                             ⊢ Membership.mem ((fun x => e) x✝).source x✝
                           -/
  mem_chart_source _ := by rw [h]; apply mem_univ
                                   /-
                                     🎉 no goals
                                   -/
                          /-
                            H : Type u
                            H' : Type u_1
                            M : Type u_2
                            M' : Type u_3
                            M'' : Type u_4
                            inst✝³ : TopologicalSpace H
                            inst✝² : TopologicalSpace M
                            inst✝¹ : ChartedSpace H M
                            α : Type u_5
                            inst✝ : TopologicalSpace α
                            e : PartialHomeomorph α H
                            h : Eq e.source Set.univ
                            x✝ : α
                            ⊢ Membership.mem (Singleton.singleton e) ((fun x => e) x✝)
                          -/
  chart_mem_atlas _ := by tauto
                          /-
                            🎉 no goals
                          -/


@[simp, mfld_simps]
theorem singletonChartedSpace_chartAt_eq (h : e.source = Set.univ) {x : α} :
    @chartAt H _ α _ (e.singletonChartedSpace h) x = e :=
  rfl


theorem singletonChartedSpace_chartAt_source (h : e.source = Set.univ) {x : α} :
    (@chartAt H _ α _ (e.singletonChartedSpace h) x).source = Set.univ :=
  h


theorem singletonChartedSpace_mem_atlas_eq (h : e.source = Set.univ) (e' : PartialHomeomorph α H)
    (h' : e' ∈ (e.singletonChartedSpace h).atlas) : e' = e :=
  h'


/-- Given a partial homeomorphism `e` from a space `α` into `H`, if its source covers the whole
space `α`, then the induced charted space structure on `α` is `HasGroupoid G` for any structure
groupoid `G` which is closed under restrictions. -/
theorem singleton_hasGroupoid (h : e.source = Set.univ) (G : StructureGroupoid H)
    [ClosedUnderRestriction G] : @HasGroupoid _ _ _ _ (e.singletonChartedSpace h) G :=
  { __ := e.singletonChartedSpace h
    compatible := by
      /-
        H : Type u
        inst✝² : TopologicalSpace H
        α : Type u_5
        inst✝¹ : TopologicalSpace α
        e : PartialHomeomorph α H
        h : Eq e.source Set.univ
        G : StructureGroupoid H
        inst✝ : ClosedUnderRestriction G
        ⊢ ∀ {e_1 e' : PartialHomeomorph α H}, Membership.mem (atlas H α) e_1 → Members …
      -/
      intro e' e'' he' he''
      rw [e.singletonChartedSpace_mem_atlas_eq h e' he',
        e.singletonChartedSpace_mem_atlas_eq h e'' he'']
      /-
        H : Type u
        inst✝² : TopologicalSpace H
        α : Type u_5
        inst✝¹ : TopologicalSpace α
        e : PartialHomeomorph α H
        h : Eq e.source Set.univ
        G : StructureGroupoid H
        inst✝ : ClosedUnderRestriction G
        e' e'' : PartialHomeomorph α H
        he' : Membership.mem (atlas H α) e'
        he'' : Membership.mem (atlas H α) e''
        ⊢ Membership.mem G (e.symm.trans e)
      -/
      refine G.mem_of_eqOnSource ?_ e.symm_trans_self
      /-
        H : Type u
        inst✝² : TopologicalSpace H
        α : Type u_5
        inst✝¹ : TopologicalSpace α
        e : PartialHomeomorph α H
        h : Eq e.source Set.univ
        G : StructureGroupoid H
        inst✝ : ClosedUnderRestriction G
        e' e'' : PartialHomeomorph α H
        he' : Membership.mem (atlas H α) e'
        he'' : Membership.mem (atlas H α) e''
        ⊢ Membership.mem G (PartialHomeomorph.ofSet e.target ⋯)
      -/
      have hle : idRestrGroupoid ≤ G := (closedUnderRestriction_iff_id_le G).mp (by assumption)
      /-
        H : Type u
        inst✝² : TopologicalSpace H
        α : Type u_5
        inst✝¹ : TopologicalSpace α
        e : PartialHomeomorph α H
        h : Eq e.source Set.univ
        G : StructureGroupoid H
        inst✝ : ClosedUnderRestriction G
        e' e'' : PartialHomeomorph α H
        he' : Membership.mem (atlas H α) e'
        he'' : Membership.mem (atlas H α) e''
        hle : LE.le idRestrGroupoid G
        ⊢ Membership.mem G (PartialHomeomorph.ofSet e.target ⋯)
      -/
      exact StructureGroupoid.le_iff.mp hle _ (idRestrGroupoid_mem _) }
      /-
        🎉 no goals
      -/


/-- An open embedding of `α` into `H` induces an `H`-charted space structure on `α`.
See `PartialHomeomorph.singletonChartedSpace`. -/
def singletonChartedSpace {f : α → H} (h : IsOpenEmbedding f) : ChartedSpace H α :=
  (h.toPartialHomeomorph f).singletonChartedSpace (toPartialHomeomorph_source _ _)


theorem singletonChartedSpace_chartAt_eq {f : α → H} (h : IsOpenEmbedding f) {x : α} :
    ⇑(@chartAt H _ α _ h.singletonChartedSpace x) = f :=
  rfl


theorem singleton_hasGroupoid {f : α → H} (h : IsOpenEmbedding f) (G : StructureGroupoid H)
    [ClosedUnderRestriction G] : @HasGroupoid _ _ _ _ h.singletonChartedSpace G :=
  (h.toPartialHomeomorph f).singleton_hasGroupoid (toPartialHomeomorph_source _ _) G


/-- An open subset of a charted space is naturally a charted space. -/
protected instance instChartedSpace : ChartedSpace H s where
  atlas := ⋃ x : s, {(chartAt H x.1).subtypeRestr ⟨x⟩}
  chartAt x := (chartAt H x.1).subtypeRestr ⟨x⟩
  mem_chart_source x := ⟨trivial, mem_chart_source H x.1⟩
  chart_mem_atlas x := by
    /-
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝³ : TopologicalSpace H
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      G : StructureGroupoid H
      inst✝ : HasGroupoid M G
      s : TopologicalSpace.Opens M
      x : Subtype fun x => Membership.mem s x
      ⊢ Membership.mem (Set.iUnion fun x => Singleton.singleton ((chartAt H ↑x).subt …
    -/
    simp only [mem_iUnion, mem_singleton_iff]
    /-
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝³ : TopologicalSpace H
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      G : StructureGroupoid H
      inst✝ : HasGroupoid M G
      s : TopologicalSpace.Opens M
      x : Subtype fun x => Membership.mem s x
      ⊢ Exists fun i => Eq ((chartAt H ↑x).subtypeRestr ⋯) ((chartAt H ↑i).subtypeRe …
    -/
    use x
    /-
      🎉 no goals
    -/


/-- If `s` is a non-empty open subset of `M`, every chart of `s` is the restriction
 of some chart on `M`. -/
lemma chart_eq {s : Opens M} (hs : Nonempty s) {e : PartialHomeomorph s H} (he : e ∈ atlas H s) :
    ∃ x : s, e = (chartAt H (x : M)).subtypeRestr hs := by
  /-
    H : Type u
    M : Type u_2
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    s : TopologicalSpace.Opens M
    hs : Nonempty (Subtype fun x => Membership.mem s x)
    e : PartialHomeomorph (Subtype fun x => Membership.mem s x) H
    he : Membership.mem (atlas H (Subtype fun x => Membership.mem s x)) e
    ⊢ Exists fun x => Eq e ((chartAt H ↑x).subtypeRestr hs)
  -/
  rcases he with ⟨xset, ⟨x, hx⟩, he⟩
  /-
    case intro.intro.intro
    H : Type u
    M : Type u_2
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    s : TopologicalSpace.Opens M
    hs : Nonempty (Subtype fun x => Membership.mem s x)
    e : PartialHomeomorph (Subtype fun x => Membership.mem s x) H
    xset : Set (PartialHomeomorph (Subtype fun x => Membership.mem s x) H)
    he : Membership.mem xset e
    x : Subtype fun x => Membership.mem s x
    hx : Eq ((fun x => Singleton.singleton ((chartAt H ↑x).subtypeRestr ⋯)) x) xset
    ⊢ Exists fun x => Eq e ((chartAt H ↑x).subtypeRestr hs)
  -/
  exact ⟨x, mem_singleton_iff.mp (by convert he)⟩
  /-
    🎉 no goals
  -/


/-- If `t` is a non-empty open subset of `H`,
  every chart of `t` is the restriction of some chart on `H`. -/
-- XXX: can I unify this with `chart_eq`?
lemma chart_eq' {t : Opens H} (ht : Nonempty t) {e' : PartialHomeomorph t H}
    (he' : e' ∈ atlas H t) : ∃ x : t, e' = (chartAt H ↑x).subtypeRestr ht := by
  /-
    H : Type u
    inst✝ : TopologicalSpace H
    t : TopologicalSpace.Opens H
    ht : Nonempty (Subtype fun x => Membership.mem t x)
    e' : PartialHomeomorph (Subtype fun x => Membership.mem t x) H
    he' : Membership.mem (atlas H (Subtype fun x => Membership.mem t x)) e'
    ⊢ Exists fun x => Eq e' ((chartAt H ↑x).subtypeRestr ht)
  -/
  rcases he' with ⟨xset, ⟨x, hx⟩, he'⟩
  /-
    case intro.intro.intro
    H : Type u
    inst✝ : TopologicalSpace H
    t : TopologicalSpace.Opens H
    ht : Nonempty (Subtype fun x => Membership.mem t x)
    e' : PartialHomeomorph (Subtype fun x => Membership.mem t x) H
    xset : Set (PartialHomeomorph (Subtype fun x => Membership.mem t x) H)
    he' : Membership.mem xset e'
    x : Subtype fun x => Membership.mem t x
    hx : Eq ((fun x => Singleton.singleton ((chartAt H ↑x).subtypeRestr ⋯)) x) xset
    ⊢ Exists fun x => Eq e' ((chartAt H ↑x).subtypeRestr ht)
  -/
  exact ⟨x, mem_singleton_iff.mp (by convert he')⟩
  /-
    🎉 no goals
  -/


/-- If a groupoid `G` is `ClosedUnderRestriction`, then an open subset of a space which is
`HasGroupoid G` is naturally `HasGroupoid G`. -/
protected instance instHasGroupoid [ClosedUnderRestriction G] : HasGroupoid s G where
  compatible := by
    /-
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝⁴ : TopologicalSpace H
      inst✝³ : TopologicalSpace M
      inst✝² : ChartedSpace H M
      G : StructureGroupoid H
      inst✝¹ : HasGroupoid M G
      s : TopologicalSpace.Opens M
      inst✝ : ClosedUnderRestriction G
      ⊢ ∀ {e e' : PartialHomeomorph (Subtype fun x => Membership.mem s x) H}, Member …
    -/
    rintro e e' ⟨_, ⟨x, hc⟩, he⟩ ⟨_, ⟨x', hc'⟩, he'⟩
    /-
      case intro.intro.intro.intro.intro.intro
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝⁴ : TopologicalSpace H
      inst✝³ : TopologicalSpace M
      inst✝² : ChartedSpace H M
      G : StructureGroupoid H
      inst✝¹ : HasGroupoid M G
      s : TopologicalSpace.Opens M
      inst✝ : ClosedUnderRestriction G
      e e' : PartialHomeomorph (Subtype fun x => Membership.mem s x) H
      w✝¹ : Set (PartialHomeomorph (Subtype fun x => Membership.mem s x) H)
      he : Membership.mem w✝¹ e
      x : Subtype fun x => Membership.mem s x
      hc : Eq ((fun x => Singleton.singleton ((chartAt H ↑x).subtypeRestr ⋯)) x) w✝¹
      w✝ : Set (PartialHomeomorph (Subtype fun x => Membership.mem s x) H)
      he' : Membership.mem w✝ e'
      x' : Subtype fun x => Membership.mem s x
      hc' : Eq ((fun x => Singleton.singleton ((chartAt H ↑x).subtypeRestr ⋯)) x') w✝
      ⊢ Membership.mem G (e.symm.trans e')
    -/
    rw [hc.symm, mem_singleton_iff] at he
    /-
      case intro.intro.intro.intro.intro.intro
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝⁴ : TopologicalSpace H
      inst✝³ : TopologicalSpace M
      inst✝² : ChartedSpace H M
      G : StructureGroupoid H
      inst✝¹ : HasGroupoid M G
      s : TopologicalSpace.Opens M
      inst✝ : ClosedUnderRestriction G
      e e' : PartialHomeomorph (Subtype fun x => Membership.mem s x) H
      w✝¹ : Set (PartialHomeomorph (Subtype fun x => Membership.mem s x) H)
      x : Subtype fun x => Membership.mem s x
      he : Eq e ((chartAt H ↑x).subtypeRestr ⋯)
      hc : Eq ((fun x => Singleton.singleton ((chartAt H ↑x).subtypeRestr ⋯)) x) w✝¹
      w✝ : Set (PartialHomeomorph (Subtype fun x => Membership.mem s x) H)
      he' : Membership.mem w✝ e'
      x' : Subtype fun x => Membership.mem s x
      hc' : Eq ((fun x => Singleton.singleton ((chartAt H ↑x).subtypeRestr ⋯)) x') w✝
      ⊢ Membership.mem G (e.symm.trans e')
    -/
    rw [hc'.symm, mem_singleton_iff] at he'
    /-
      case intro.intro.intro.intro.intro.intro
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝⁴ : TopologicalSpace H
      inst✝³ : TopologicalSpace M
      inst✝² : ChartedSpace H M
      G : StructureGroupoid H
      inst✝¹ : HasGroupoid M G
      s : TopologicalSpace.Opens M
      inst✝ : ClosedUnderRestriction G
      e e' : PartialHomeomorph (Subtype fun x => Membership.mem s x) H
      w✝¹ : Set (PartialHomeomorph (Subtype fun x => Membership.mem s x) H)
      x : Subtype fun x => Membership.mem s x
      he : Eq e ((chartAt H ↑x).subtypeRestr ⋯)
      hc : Eq ((fun x => Singleton.singleton ((chartAt H ↑x).subtypeRestr ⋯)) x) w✝¹
      w✝ : Set (PartialHomeomorph (Subtype fun x => Membership.mem s x) H)
      x' : Subtype fun x => Membership.mem s x
      he' : Eq e' ((chartAt H ↑x').subtypeRestr ⋯)
      hc' : Eq ((fun x => Singleton.singleton ((chartAt H ↑x).subtypeRestr ⋯)) x') w✝
      ⊢ Membership.mem G (e.symm.trans e')
    -/
    rw [he, he']
    refine G.mem_of_eqOnSource ?_
      (subtypeRestr_symm_trans_subtypeRestr (s := s) _ (chartAt H x) (chartAt H x'))
    /-
      case intro.intro.intro.intro.intro.intro
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝⁴ : TopologicalSpace H
      inst✝³ : TopologicalSpace M
      inst✝² : ChartedSpace H M
      G : StructureGroupoid H
      inst✝¹ : HasGroupoid M G
      s : TopologicalSpace.Opens M
      inst✝ : ClosedUnderRestriction G
      e e' : PartialHomeomorph (Subtype fun x => Membership.mem s x) H
      w✝¹ : Set (PartialHomeomorph (Subtype fun x => Membership.mem s x) H)
      x : Subtype fun x => Membership.mem s x
      he : Eq e ((chartAt H ↑x).subtypeRestr ⋯)
      hc : Eq ((fun x => Singleton.singleton ((chartAt H ↑x).subtypeRestr ⋯)) x) w✝¹
      w✝ : Set (PartialHomeomorph (Subtype fun x => Membership.mem s x) H)
      x' : Subtype fun x => Membership.mem s x
      he' : Eq e' ((chartAt H ↑x').subtypeRestr ⋯)
      hc' : Eq ((fun x => Singleton.singleton ((chartAt H ↑x).subtypeRestr ⋯)) x') w✝
      ⊢ Membership.mem G (((chartAt H ↑x).symm.trans (chartAt H ↑x')).restr (Inter.i …
    -/
    apply closedUnderRestriction'
      /-
        case intro.intro.intro.intro.intro.intro.he
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝⁴ : TopologicalSpace H
        inst✝³ : TopologicalSpace M
        inst✝² : ChartedSpace H M
        G : StructureGroupoid H
        inst✝¹ : HasGroupoid M G
        s : TopologicalSpace.Opens M
        inst✝ : ClosedUnderRestriction G
        e e' : PartialHomeomorph (Subtype fun x => Membership.mem s x) H
        w✝¹ : Set (PartialHomeomorph (Subtype fun x => Membership.mem s x) H)
        x : Subtype fun x => Membership.mem s x
        he : Eq e ((chartAt H ↑x).subtypeRestr ⋯)
        hc : Eq ((fun x => Singleton.singleton ((chartAt H ↑x).subtypeRestr ⋯)) x) w✝¹
        w✝ : Set (PartialHomeomorph (Subtype fun x => Membership.mem s x) H)
        x' : Subtype fun x => Membership.mem s x
        he' : Eq e' ((chartAt H ↑x').subtypeRestr ⋯)
        hc' : Eq ((fun x => Singleton.singleton ((chartAt H ↑x).subtypeRestr ⋯)) x') w✝
        ⊢ Membership.mem G ((chartAt H ↑x).symm.trans (chartAt H ↑x'))
      -/
    · exact G.compatible (chart_mem_atlas _ _) (chart_mem_atlas _ _)
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.intro.intro.hs
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝⁴ : TopologicalSpace H
        inst✝³ : TopologicalSpace M
        inst✝² : ChartedSpace H M
        G : StructureGroupoid H
        inst✝¹ : HasGroupoid M G
        s : TopologicalSpace.Opens M
        inst✝ : ClosedUnderRestriction G
        e e' : PartialHomeomorph (Subtype fun x => Membership.mem s x) H
        w✝¹ : Set (PartialHomeomorph (Subtype fun x => Membership.mem s x) H)
        x : Subtype fun x => Membership.mem s x
        he : Eq e ((chartAt H ↑x).subtypeRestr ⋯)
        hc : Eq ((fun x => Singleton.singleton ((chartAt H ↑x).subtypeRestr ⋯)) x) w✝¹
        w✝ : Set (PartialHomeomorph (Subtype fun x => Membership.mem s x) H)
        x' : Subtype fun x => Membership.mem s x
        he' : Eq e' ((chartAt H ↑x').subtypeRestr ⋯)
        hc' : Eq ((fun x => Singleton.singleton ((chartAt H ↑x).subtypeRestr ⋯)) x') w✝
        ⊢ IsOpen (Inter.inter (chartAt H ↑x).target (Set.preimage ↑(chartAt H ↑x).symm …
      -/
    · exact isOpen_inter_preimage_symm (chartAt _ _) s.2
      /-
        🎉 no goals
      -/


theorem chartAt_subtype_val_symm_eventuallyEq (U : Opens M) {x : U} :
    (chartAt H x.val).symm =ᶠ[𝓝 (chartAt H x.val x.val)] Subtype.val ∘ (chartAt H x).symm := by
  /-
    H : Type u
    M : Type u_2
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    U : TopologicalSpace.Opens M
    x : Subtype fun x => Membership.mem U x
    ⊢ (nhds (↑(chartAt H ↑x) ↑x)).EventuallyEq (↑(chartAt H ↑x).symm) (Function.co …
  -/
  set e := chartAt H x.val
  have heUx_nhds : (e.subtypeRestr ⟨x⟩).target ∈ 𝓝 (e x) := by
    apply (e.subtypeRestr ⟨x⟩).open_target.mem_nhds
    exact e.map_subtype_source ⟨x⟩ (mem_chart_source _ _)
  /-
    H : Type u
    M : Type u_2
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    U : TopologicalSpace.Opens M
    x : Subtype fun x => Membership.mem U x
    e : PartialHomeomorph M H := chartAt H ↑x
    heUx_nhds : Membership.mem (nhds (↑e ↑x)) (e.subtypeRestr ⋯).target
    ⊢ (nhds (↑e ↑x)).EventuallyEq (↑e.symm) (Function.comp Subtype.val ↑(chartAt H …
  -/
  exact Filter.eventuallyEq_of_mem heUx_nhds (e.subtypeRestr_symm_eqOn ⟨x⟩)
  /-
    🎉 no goals
  -/


theorem chartAt_inclusion_symm_eventuallyEq {U V : Opens M} (hUV : U ≤ V) {x : U} :
    (chartAt H (Opens.inclusion hUV x)).symm
    =ᶠ[𝓝 (chartAt H (Opens.inclusion hUV x) (Set.inclusion hUV x))]
    Opens.inclusion hUV ∘ (chartAt H x).symm := by
  /-
    H : Type u
    M : Type u_2
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    U V : TopologicalSpace.Opens M
    hUV : LE.le U V
    x : Subtype fun x => Membership.mem U x
    ⊢ (nhds (↑(chartAt H (TopologicalSpace.Opens.inclusion hUV x)) (Set.inclusion  …
  -/
  set e := chartAt H (x : M)
  have heUx_nhds : (e.subtypeRestr ⟨x⟩).target ∈ 𝓝 (e x) := by
    apply (e.subtypeRestr ⟨x⟩).open_target.mem_nhds
    exact e.map_subtype_source ⟨x⟩ (mem_chart_source _ _)
  exact Filter.eventuallyEq_of_mem heUx_nhds <| e.subtypeRestr_symm_eqOn_of_le ⟨x⟩
    ⟨Opens.inclusion hUV x⟩ hUV

/-- Restricting a chart of `M` to an open subset `s` yields a chart in the maximal atlas of `s`.

NB. We cannot deduce membership in `atlas H s` in general: by definition, this atlas contains
precisely the restriction of each preferred chart at `x ∈ s` --- whereas `atlas H M`
can contain more charts than these. -/
lemma StructureGroupoid.restriction_in_maximalAtlas {e : PartialHomeomorph M H}
    (he : e ∈ atlas H M) {s : Opens M} (hs : Nonempty s) {G : StructureGroupoid H} [HasGroupoid M G]
    [ClosedUnderRestriction G] : e.subtypeRestr hs ∈ G.maximalAtlas s := by
  /-
    H : Type u
    M : Type u_2
    inst✝⁴ : TopologicalSpace H
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    e : PartialHomeomorph M H
    he : Membership.mem (atlas H M) e
    s : TopologicalSpace.Opens M
    hs : Nonempty (Subtype fun x => Membership.mem s x)
    G : StructureGroupoid H
    inst✝¹ : HasGroupoid M G
    inst✝ : ClosedUnderRestriction G
    ⊢ Membership.mem (StructureGroupoid.maximalAtlas (Subtype fun x => Membership. …
  -/
  intro e' he'
  -- `e'` is the restriction of some chart of `M` at `x`,
  /-
    H : Type u
    M : Type u_2
    inst✝⁴ : TopologicalSpace H
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    e : PartialHomeomorph M H
    he : Membership.mem (atlas H M) e
    s : TopologicalSpace.Opens M
    hs : Nonempty (Subtype fun x => Membership.mem s x)
    G : StructureGroupoid H
    inst✝¹ : HasGroupoid M G
    inst✝ : ClosedUnderRestriction G
    e' : PartialHomeomorph (Subtype fun x => Membership.mem s x) H
    he' : Membership.mem (atlas H (Subtype fun x => Membership.mem s x)) e'
    ⊢ And (Membership.mem G ((e.subtypeRestr hs).symm.trans e')) (Membership.mem G …
  -/
  obtain ⟨x, this⟩ := Opens.chart_eq hs he'
  /-
    case intro
    H : Type u
    M : Type u_2
    inst✝⁴ : TopologicalSpace H
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    e : PartialHomeomorph M H
    he : Membership.mem (atlas H M) e
    s : TopologicalSpace.Opens M
    hs : Nonempty (Subtype fun x => Membership.mem s x)
    G : StructureGroupoid H
    inst✝¹ : HasGroupoid M G
    inst✝ : ClosedUnderRestriction G
    e' : PartialHomeomorph (Subtype fun x => Membership.mem s x) H
    he' : Membership.mem (atlas H (Subtype fun x => Membership.mem s x)) e'
    x : Subtype fun x => Membership.mem s x
    this : Eq e' ((chartAt H ↑x).subtypeRestr hs)
    ⊢ And (Membership.mem G ((e.subtypeRestr hs).symm.trans e')) (Membership.mem G …
  -/
  rw [this]
  -- The transition functions between the unrestricted charts lie in the groupoid,
  -- the transition functions of the restriction are the restriction of the transition function.
  exact ⟨G.trans_restricted he (chart_mem_atlas H (x : M)) hs,
         G.trans_restricted (chart_mem_atlas H (x : M)) he hs⟩


/-- A `G`-diffeomorphism between two charted spaces is a homeomorphism which, when read in the
charts, belongs to `G`. We avoid the word diffeomorph as it is too related to the smooth category,
and use structomorph instead. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): this linter isn't ported yet.
-- @[nolint has_nonempty_instance]
structure Structomorph (G : StructureGroupoid H) (M : Type*) (M' : Type*) [TopologicalSpace M]
  [TopologicalSpace M'] [ChartedSpace H M] [ChartedSpace H M'] extends Homeomorph M M' where
  mem_groupoid : ∀ c : PartialHomeomorph M H, ∀ c' : PartialHomeomorph M' H, c ∈ atlas H M →
    c' ∈ atlas H M' → c.symm ≫ₕ toHomeomorph.toPartialHomeomorph ≫ₕ c' ∈ G


/-- The identity is a diffeomorphism of any charted space, for any groupoid. -/
def Structomorph.refl (M : Type*) [TopologicalSpace M] [ChartedSpace H M] [HasGroupoid M G] :
    Structomorph G M M :=
  { Homeomorph.refl M with
    mem_groupoid := fun c c' hc hc' ↦ by
      /-
        H : Type u
        H' : Type u_1
        M✝ : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝⁹ : TopologicalSpace H
        inst✝⁸ : TopologicalSpace M✝
        inst✝⁷ : ChartedSpace H M✝
        inst✝⁶ : TopologicalSpace M'
        inst✝⁵ : TopologicalSpace M''
        G : StructureGroupoid H
        inst✝⁴ : ChartedSpace H M'
        inst✝³ : ChartedSpace H M''
        M : Type u_5
        inst✝² : TopologicalSpace M
        inst✝¹ : ChartedSpace H M
        inst✝ : HasGroupoid M G
        c c' : PartialHomeomorph M H
        hc : Membership.mem (atlas H M) c
        hc' : Membership.mem (atlas H M) c'
        ⊢ Membership.mem G (c.symm.trans (__src✝.toPartialHomeomorph.trans c'))
      -/
      change PartialHomeomorph.symm c ≫ₕ PartialHomeomorph.refl M ≫ₕ c' ∈ G
      /-
        H : Type u
        H' : Type u_1
        M✝ : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝⁹ : TopologicalSpace H
        inst✝⁸ : TopologicalSpace M✝
        inst✝⁷ : ChartedSpace H M✝
        inst✝⁶ : TopologicalSpace M'
        inst✝⁵ : TopologicalSpace M''
        G : StructureGroupoid H
        inst✝⁴ : ChartedSpace H M'
        inst✝³ : ChartedSpace H M''
        M : Type u_5
        inst✝² : TopologicalSpace M
        inst✝¹ : ChartedSpace H M
        inst✝ : HasGroupoid M G
        c c' : PartialHomeomorph M H
        hc : Membership.mem (atlas H M) c
        hc' : Membership.mem (atlas H M) c'
        ⊢ Membership.mem G (c.symm.trans ((PartialHomeomorph.refl M).trans c'))
      -/
      rw [PartialHomeomorph.refl_trans]
      /-
        H : Type u
        H' : Type u_1
        M✝ : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝⁹ : TopologicalSpace H
        inst✝⁸ : TopologicalSpace M✝
        inst✝⁷ : ChartedSpace H M✝
        inst✝⁶ : TopologicalSpace M'
        inst✝⁵ : TopologicalSpace M''
        G : StructureGroupoid H
        inst✝⁴ : ChartedSpace H M'
        inst✝³ : ChartedSpace H M''
        M : Type u_5
        inst✝² : TopologicalSpace M
        inst✝¹ : ChartedSpace H M
        inst✝ : HasGroupoid M G
        c c' : PartialHomeomorph M H
        hc : Membership.mem (atlas H M) c
        hc' : Membership.mem (atlas H M) c'
        ⊢ Membership.mem G (c.symm.trans c')
      -/
      exact G.compatible hc hc' }
      /-
        🎉 no goals
      -/


/-- The inverse of a structomorphism is a structomorphism. -/
def Structomorph.symm (e : Structomorph G M M') : Structomorph G M' M :=
  { e.toHomeomorph.symm with
    mem_groupoid := by
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝⁶ : TopologicalSpace H
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : ChartedSpace H M
        inst✝³ : TopologicalSpace M'
        inst✝² : TopologicalSpace M''
        G : StructureGroupoid H
        inst✝¹ : ChartedSpace H M'
        inst✝ : ChartedSpace H M''
        e : Structomorph G M M'
        ⊢ ∀ (c : PartialHomeomorph M' H) (c' : PartialHomeomorph M H), Membership.mem  …
      -/
      intro c c' hc hc'
      have : (c'.symm ≫ₕ e.toHomeomorph.toPartialHomeomorph ≫ₕ c).symm ∈ G :=
        G.symm (e.mem_groupoid c' c hc' hc)
      rwa [trans_symm_eq_symm_trans_symm, trans_symm_eq_symm_trans_symm, symm_symm, trans_assoc]
        at this }


/-- The composition of structomorphisms is a structomorphism. -/
def Structomorph.trans (e : Structomorph G M M') (e' : Structomorph G M' M'') :
    Structomorph G M M'' :=
  { Homeomorph.trans e.toHomeomorph e'.toHomeomorph with
    mem_groupoid := by
      /- Let c and c' be two charts in M and M''. We want to show that e' ∘ e is smooth in these
      charts, around any point x. For this, let y = e (c⁻¹ x), and consider a chart g around y.
      Then g ∘ e ∘ c⁻¹ and c' ∘ e' ∘ g⁻¹ are both smooth as e and e' are structomorphisms, so
      their composition is smooth, and it coincides with c' ∘ e' ∘ e ∘ c⁻¹ around x. -/
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝⁶ : TopologicalSpace H
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : ChartedSpace H M
        inst✝³ : TopologicalSpace M'
        inst✝² : TopologicalSpace M''
        G : StructureGroupoid H
        inst✝¹ : ChartedSpace H M'
        inst✝ : ChartedSpace H M''
        e : Structomorph G M M'
        e' : Structomorph G M' M''
        ⊢ ∀ (c : PartialHomeomorph M H) (c' : PartialHomeomorph M'' H), Membership.mem …
      -/
      intro c c' hc hc'
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝⁶ : TopologicalSpace H
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : ChartedSpace H M
        inst✝³ : TopologicalSpace M'
        inst✝² : TopologicalSpace M''
        G : StructureGroupoid H
        inst✝¹ : ChartedSpace H M'
        inst✝ : ChartedSpace H M''
        e : Structomorph G M M'
        e' : Structomorph G M' M''
        c : PartialHomeomorph M H
        c' : PartialHomeomorph M'' H
        hc : Membership.mem (atlas H M) c
        hc' : Membership.mem (atlas H M'') c'
        ⊢ Membership.mem G (c.symm.trans (__src✝.toPartialHomeomorph.trans c'))
      -/
      refine G.locality fun x hx ↦ ?_
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝⁶ : TopologicalSpace H
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : ChartedSpace H M
        inst✝³ : TopologicalSpace M'
        inst✝² : TopologicalSpace M''
        G : StructureGroupoid H
        inst✝¹ : ChartedSpace H M'
        inst✝ : ChartedSpace H M''
        e : Structomorph G M M'
        e' : Structomorph G M' M''
        c : PartialHomeomorph M H
        c' : PartialHomeomorph M'' H
        hc : Membership.mem (atlas H M) c
        hc' : Membership.mem (atlas H M'') c'
        x : H
        hx : Membership.mem (c.symm.trans (__src✝.toPartialHomeomorph.trans c')).sourc …
        ⊢ Exists fun s => And (IsOpen s) (And (Membership.mem s x) (Membership.mem G ( …
      -/
      let f₁ := e.toHomeomorph.toPartialHomeomorph
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝⁶ : TopologicalSpace H
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : ChartedSpace H M
        inst✝³ : TopologicalSpace M'
        inst✝² : TopologicalSpace M''
        G : StructureGroupoid H
        inst✝¹ : ChartedSpace H M'
        inst✝ : ChartedSpace H M''
        e : Structomorph G M M'
        e' : Structomorph G M' M''
        c : PartialHomeomorph M H
        c' : PartialHomeomorph M'' H
        hc : Membership.mem (atlas H M) c
        hc' : Membership.mem (atlas H M'') c'
        x : H
        hx : Membership.mem (c.symm.trans (__src✝.toPartialHomeomorph.trans c')).sourc …
        f₁ : PartialHomeomorph M M' := e.toPartialHomeomorph
        ⊢ Exists fun s => And (IsOpen s) (And (Membership.mem s x) (Membership.mem G ( …
      -/
      let f₂ := e'.toHomeomorph.toPartialHomeomorph
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝⁶ : TopologicalSpace H
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : ChartedSpace H M
        inst✝³ : TopologicalSpace M'
        inst✝² : TopologicalSpace M''
        G : StructureGroupoid H
        inst✝¹ : ChartedSpace H M'
        inst✝ : ChartedSpace H M''
        e : Structomorph G M M'
        e' : Structomorph G M' M''
        c : PartialHomeomorph M H
        c' : PartialHomeomorph M'' H
        hc : Membership.mem (atlas H M) c
        hc' : Membership.mem (atlas H M'') c'
        x : H
        hx : Membership.mem (c.symm.trans (__src✝.toPartialHomeomorph.trans c')).sourc …
        f₁ : PartialHomeomorph M M' := e.toPartialHomeomorph
        f₂ : PartialHomeomorph M' M'' := e'.toPartialHomeomorph
        ⊢ Exists fun s => And (IsOpen s) (And (Membership.mem s x) (Membership.mem G ( …
      -/
      let f := (e.toHomeomorph.trans e'.toHomeomorph).toPartialHomeomorph
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝⁶ : TopologicalSpace H
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : ChartedSpace H M
        inst✝³ : TopologicalSpace M'
        inst✝² : TopologicalSpace M''
        G : StructureGroupoid H
        inst✝¹ : ChartedSpace H M'
        inst✝ : ChartedSpace H M''
        e : Structomorph G M M'
        e' : Structomorph G M' M''
        c : PartialHomeomorph M H
        c' : PartialHomeomorph M'' H
        hc : Membership.mem (atlas H M) c
        hc' : Membership.mem (atlas H M'') c'
        x : H
        hx : Membership.mem (c.symm.trans (__src✝.toPartialHomeomorph.trans c')).sourc …
        f₁ : PartialHomeomorph M M' := e.toPartialHomeomorph
        f₂ : PartialHomeomorph M' M'' := e'.toPartialHomeomorph
        f : PartialHomeomorph M M'' := (e.trans e'.toHomeomorph).toPartialHomeomorph
        ⊢ Exists fun s => And (IsOpen s) (And (Membership.mem s x) (Membership.mem G ( …
      -/
      have feq : f = f₁ ≫ₕ f₂ := Homeomorph.trans_toPartialHomeomorph _ _
      -- define the atlas g around y
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝⁶ : TopologicalSpace H
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : ChartedSpace H M
        inst✝³ : TopologicalSpace M'
        inst✝² : TopologicalSpace M''
        G : StructureGroupoid H
        inst✝¹ : ChartedSpace H M'
        inst✝ : ChartedSpace H M''
        e : Structomorph G M M'
        e' : Structomorph G M' M''
        c : PartialHomeomorph M H
        c' : PartialHomeomorph M'' H
        hc : Membership.mem (atlas H M) c
        hc' : Membership.mem (atlas H M'') c'
        x : H
        hx : Membership.mem (c.symm.trans (__src✝.toPartialHomeomorph.trans c')).sourc …
        f₁ : PartialHomeomorph M M' := e.toPartialHomeomorph
        f₂ : PartialHomeomorph M' M'' := e'.toPartialHomeomorph
        f : PartialHomeomorph M M'' := (e.trans e'.toHomeomorph).toPartialHomeomorph
        feq : Eq f (f₁.trans f₂)
        ⊢ Exists fun s => And (IsOpen s) (And (Membership.mem s x) (Membership.mem G ( …
      -/
      let y := (c.symm ≫ₕ f₁) x
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝⁶ : TopologicalSpace H
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : ChartedSpace H M
        inst✝³ : TopologicalSpace M'
        inst✝² : TopologicalSpace M''
        G : StructureGroupoid H
        inst✝¹ : ChartedSpace H M'
        inst✝ : ChartedSpace H M''
        e : Structomorph G M M'
        e' : Structomorph G M' M''
        c : PartialHomeomorph M H
        c' : PartialHomeomorph M'' H
        hc : Membership.mem (atlas H M) c
        hc' : Membership.mem (atlas H M'') c'
        x : H
        hx : Membership.mem (c.symm.trans (__src✝.toPartialHomeomorph.trans c')).sourc …
        f₁ : PartialHomeomorph M M' := e.toPartialHomeomorph
        f₂ : PartialHomeomorph M' M'' := e'.toPartialHomeomorph
        f : PartialHomeomorph M M'' := (e.trans e'.toHomeomorph).toPartialHomeomorph
        feq : Eq f (f₁.trans f₂)
        y : M' := ↑(c.symm.trans f₁) x
        ⊢ Exists fun s => And (IsOpen s) (And (Membership.mem s x) (Membership.mem G ( …
      -/
      let g := chartAt (H := H) y
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝⁶ : TopologicalSpace H
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : ChartedSpace H M
        inst✝³ : TopologicalSpace M'
        inst✝² : TopologicalSpace M''
        G : StructureGroupoid H
        inst✝¹ : ChartedSpace H M'
        inst✝ : ChartedSpace H M''
        e : Structomorph G M M'
        e' : Structomorph G M' M''
        c : PartialHomeomorph M H
        c' : PartialHomeomorph M'' H
        hc : Membership.mem (atlas H M) c
        hc' : Membership.mem (atlas H M'') c'
        x : H
        hx : Membership.mem (c.symm.trans (__src✝.toPartialHomeomorph.trans c')).sourc …
        f₁ : PartialHomeomorph M M' := e.toPartialHomeomorph
        f₂ : PartialHomeomorph M' M'' := e'.toPartialHomeomorph
        f : PartialHomeomorph M M'' := (e.trans e'.toHomeomorph).toPartialHomeomorph
        feq : Eq f (f₁.trans f₂)
        y : M' := ↑(c.symm.trans f₁) x
        g : PartialHomeomorph M' H := chartAt H y
        ⊢ Exists fun s => And (IsOpen s) (And (Membership.mem s x) (Membership.mem G ( …
      -/
      have hg₁ := chart_mem_atlas (H := H) y
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝⁶ : TopologicalSpace H
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : ChartedSpace H M
        inst✝³ : TopologicalSpace M'
        inst✝² : TopologicalSpace M''
        G : StructureGroupoid H
        inst✝¹ : ChartedSpace H M'
        inst✝ : ChartedSpace H M''
        e : Structomorph G M M'
        e' : Structomorph G M' M''
        c : PartialHomeomorph M H
        c' : PartialHomeomorph M'' H
        hc : Membership.mem (atlas H M) c
        hc' : Membership.mem (atlas H M'') c'
        x : H
        hx : Membership.mem (c.symm.trans (__src✝.toPartialHomeomorph.trans c')).sourc …
        f₁ : PartialHomeomorph M M' := e.toPartialHomeomorph
        f₂ : PartialHomeomorph M' M'' := e'.toPartialHomeomorph
        f : PartialHomeomorph M M'' := (e.trans e'.toHomeomorph).toPartialHomeomorph
        feq : Eq f (f₁.trans f₂)
        y : M' := ↑(c.symm.trans f₁) x
        g : PartialHomeomorph M' H := chartAt H y
        hg₁ : Membership.mem (atlas H M') (chartAt H y)
        ⊢ Exists fun s => And (IsOpen s) (And (Membership.mem s x) (Membership.mem G ( …
      -/
      have hg₂ := mem_chart_source (H := H) y
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝⁶ : TopologicalSpace H
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : ChartedSpace H M
        inst✝³ : TopologicalSpace M'
        inst✝² : TopologicalSpace M''
        G : StructureGroupoid H
        inst✝¹ : ChartedSpace H M'
        inst✝ : ChartedSpace H M''
        e : Structomorph G M M'
        e' : Structomorph G M' M''
        c : PartialHomeomorph M H
        c' : PartialHomeomorph M'' H
        hc : Membership.mem (atlas H M) c
        hc' : Membership.mem (atlas H M'') c'
        x : H
        hx : Membership.mem (c.symm.trans (__src✝.toPartialHomeomorph.trans c')).sourc …
        f₁ : PartialHomeomorph M M' := e.toPartialHomeomorph
        f₂ : PartialHomeomorph M' M'' := e'.toPartialHomeomorph
        f : PartialHomeomorph M M'' := (e.trans e'.toHomeomorph).toPartialHomeomorph
        feq : Eq f (f₁.trans f₂)
        y : M' := ↑(c.symm.trans f₁) x
        g : PartialHomeomorph M' H := chartAt H y
        hg₁ : Membership.mem (atlas H M') (chartAt H y)
        hg₂ : Membership.mem (chartAt H y).source y
        ⊢ Exists fun s => And (IsOpen s) (And (Membership.mem s x) (Membership.mem G ( …
      -/
      let s := (c.symm ≫ₕ f₁).source ∩ c.symm ≫ₕ f₁ ⁻¹' g.source
      have open_s : IsOpen s := by
        apply (c.symm ≫ₕ f₁).continuousOn_toFun.isOpen_inter_preimage <;> apply open_source
      have : x ∈ s := by
        constructor
        · simp only [f₁, trans_source, preimage_univ, inter_univ,
            Homeomorph.toPartialHomeomorph_source]
          rw [trans_source] at hx
          exact hx.1
        · exact hg₂
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝⁶ : TopologicalSpace H
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : ChartedSpace H M
        inst✝³ : TopologicalSpace M'
        inst✝² : TopologicalSpace M''
        G : StructureGroupoid H
        inst✝¹ : ChartedSpace H M'
        inst✝ : ChartedSpace H M''
        e : Structomorph G M M'
        e' : Structomorph G M' M''
        c : PartialHomeomorph M H
        c' : PartialHomeomorph M'' H
        hc : Membership.mem (atlas H M) c
        hc' : Membership.mem (atlas H M'') c'
        x : H
        hx : Membership.mem (c.symm.trans (__src✝.toPartialHomeomorph.trans c')).sourc …
        f₁ : PartialHomeomorph M M' := e.toPartialHomeomorph
        f₂ : PartialHomeomorph M' M'' := e'.toPartialHomeomorph
        f : PartialHomeomorph M M'' := (e.trans e'.toHomeomorph).toPartialHomeomorph
        feq : Eq f (f₁.trans f₂)
        y : M' := ↑(c.symm.trans f₁) x
        g : PartialHomeomorph M' H := chartAt H y
        hg₁ : Membership.mem (atlas H M') (chartAt H y)
        hg₂ : Membership.mem (chartAt H y).source y
        s : Set H := Inter.inter (c.symm.trans f₁).source (Set.preimage (↑(c.symm.tran …
        open_s : IsOpen s
        this : Membership.mem s x
        ⊢ Exists fun s => And (IsOpen s) (And (Membership.mem s x) (Membership.mem G ( …
      -/
      refine ⟨s, open_s, this, ?_⟩
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝⁶ : TopologicalSpace H
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : ChartedSpace H M
        inst✝³ : TopologicalSpace M'
        inst✝² : TopologicalSpace M''
        G : StructureGroupoid H
        inst✝¹ : ChartedSpace H M'
        inst✝ : ChartedSpace H M''
        e : Structomorph G M M'
        e' : Structomorph G M' M''
        c : PartialHomeomorph M H
        c' : PartialHomeomorph M'' H
        hc : Membership.mem (atlas H M) c
        hc' : Membership.mem (atlas H M'') c'
        x : H
        hx : Membership.mem (c.symm.trans (__src✝.toPartialHomeomorph.trans c')).sourc …
        f₁ : PartialHomeomorph M M' := e.toPartialHomeomorph
        f₂ : PartialHomeomorph M' M'' := e'.toPartialHomeomorph
        f : PartialHomeomorph M M'' := (e.trans e'.toHomeomorph).toPartialHomeomorph
        feq : Eq f (f₁.trans f₂)
        y : M' := ↑(c.symm.trans f₁) x
        g : PartialHomeomorph M' H := chartAt H y
        hg₁ : Membership.mem (atlas H M') (chartAt H y)
        hg₂ : Membership.mem (chartAt H y).source y
        s : Set H := Inter.inter (c.symm.trans f₁).source (Set.preimage (↑(c.symm.tran …
        open_s : IsOpen s
        this : Membership.mem s x
        ⊢ Membership.mem G ((c.symm.trans (__src✝.toPartialHomeomorph.trans c')).restr …
      -/
      let F₁ := (c.symm ≫ₕ f₁ ≫ₕ g) ≫ₕ g.symm ≫ₕ f₂ ≫ₕ c'
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝⁶ : TopologicalSpace H
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : ChartedSpace H M
        inst✝³ : TopologicalSpace M'
        inst✝² : TopologicalSpace M''
        G : StructureGroupoid H
        inst✝¹ : ChartedSpace H M'
        inst✝ : ChartedSpace H M''
        e : Structomorph G M M'
        e' : Structomorph G M' M''
        c : PartialHomeomorph M H
        c' : PartialHomeomorph M'' H
        hc : Membership.mem (atlas H M) c
        hc' : Membership.mem (atlas H M'') c'
        x : H
        hx : Membership.mem (c.symm.trans (__src✝.toPartialHomeomorph.trans c')).sourc …
        f₁ : PartialHomeomorph M M' := e.toPartialHomeomorph
        f₂ : PartialHomeomorph M' M'' := e'.toPartialHomeomorph
        f : PartialHomeomorph M M'' := (e.trans e'.toHomeomorph).toPartialHomeomorph
        feq : Eq f (f₁.trans f₂)
        y : M' := ↑(c.symm.trans f₁) x
        g : PartialHomeomorph M' H := chartAt H y
        hg₁ : Membership.mem (atlas H M') (chartAt H y)
        hg₂ : Membership.mem (chartAt H y).source y
        s : Set H := Inter.inter (c.symm.trans f₁).source (Set.preimage (↑(c.symm.tran …
        open_s : IsOpen s
        this : Membership.mem s x
        F₁ : PartialHomeomorph H H := (c.symm.trans (f₁.trans g)).trans (g.symm.trans  …
        ⊢ Membership.mem G ((c.symm.trans (__src✝.toPartialHomeomorph.trans c')).restr …
      -/
      have A : F₁ ∈ G := G.trans (e.mem_groupoid c g hc hg₁) (e'.mem_groupoid g c' hg₁ hc')
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝⁶ : TopologicalSpace H
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : ChartedSpace H M
        inst✝³ : TopologicalSpace M'
        inst✝² : TopologicalSpace M''
        G : StructureGroupoid H
        inst✝¹ : ChartedSpace H M'
        inst✝ : ChartedSpace H M''
        e : Structomorph G M M'
        e' : Structomorph G M' M''
        c : PartialHomeomorph M H
        c' : PartialHomeomorph M'' H
        hc : Membership.mem (atlas H M) c
        hc' : Membership.mem (atlas H M'') c'
        x : H
        hx : Membership.mem (c.symm.trans (__src✝.toPartialHomeomorph.trans c')).sourc …
        f₁ : PartialHomeomorph M M' := e.toPartialHomeomorph
        f₂ : PartialHomeomorph M' M'' := e'.toPartialHomeomorph
        f : PartialHomeomorph M M'' := (e.trans e'.toHomeomorph).toPartialHomeomorph
        feq : Eq f (f₁.trans f₂)
        y : M' := ↑(c.symm.trans f₁) x
        g : PartialHomeomorph M' H := chartAt H y
        hg₁ : Membership.mem (atlas H M') (chartAt H y)
        hg₂ : Membership.mem (chartAt H y).source y
        s : Set H := Inter.inter (c.symm.trans f₁).source (Set.preimage (↑(c.symm.tran …
        open_s : IsOpen s
        this : Membership.mem s x
        F₁ : PartialHomeomorph H H := (c.symm.trans (f₁.trans g)).trans (g.symm.trans  …
        A : Membership.mem G F₁
        ⊢ Membership.mem G ((c.symm.trans (__src✝.toPartialHomeomorph.trans c')).restr …
      -/
      let F₂ := (c.symm ≫ₕ f ≫ₕ c').restr s
      have : F₁ ≈ F₂ := calc
        F₁ ≈ c.symm ≫ₕ f₁ ≫ₕ (g ≫ₕ g.symm) ≫ₕ f₂ ≫ₕ c' := by
            simp only [F₁, trans_assoc, _root_.refl]
        _ ≈ c.symm ≫ₕ f₁ ≫ₕ ofSet g.source g.open_source ≫ₕ f₂ ≫ₕ c' :=
          EqOnSource.trans' (_root_.refl _) (EqOnSource.trans' (_root_.refl _)
            (EqOnSource.trans' (self_trans_symm g) (_root_.refl _)))
        _ ≈ ((c.symm ≫ₕ f₁) ≫ₕ ofSet g.source g.open_source) ≫ₕ f₂ ≫ₕ c' := by
          simp only [trans_assoc, _root_.refl]
        _ ≈ (c.symm ≫ₕ f₁).restr s ≫ₕ f₂ ≫ₕ c' := by rw [trans_of_set']
        _ ≈ ((c.symm ≫ₕ f₁) ≫ₕ f₂ ≫ₕ c').restr s := by rw [restr_trans]
        _ ≈ (c.symm ≫ₕ (f₁ ≫ₕ f₂) ≫ₕ c').restr s := by
          simp only [EqOnSource.restr, trans_assoc, _root_.refl]
        _ ≈ F₂ := by simp only [F₂, feq, _root_.refl]
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝⁶ : TopologicalSpace H
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : ChartedSpace H M
        inst✝³ : TopologicalSpace M'
        inst✝² : TopologicalSpace M''
        G : StructureGroupoid H
        inst✝¹ : ChartedSpace H M'
        inst✝ : ChartedSpace H M''
        e : Structomorph G M M'
        e' : Structomorph G M' M''
        c : PartialHomeomorph M H
        c' : PartialHomeomorph M'' H
        hc : Membership.mem (atlas H M) c
        hc' : Membership.mem (atlas H M'') c'
        x : H
        hx : Membership.mem (c.symm.trans (__src✝.toPartialHomeomorph.trans c')).sourc …
        f₁ : PartialHomeomorph M M' := e.toPartialHomeomorph
        f₂ : PartialHomeomorph M' M'' := e'.toPartialHomeomorph
        f : PartialHomeomorph M M'' := (e.trans e'.toHomeomorph).toPartialHomeomorph
        feq : Eq f (f₁.trans f₂)
        y : M' := ↑(c.symm.trans f₁) x
        g : PartialHomeomorph M' H := chartAt H y
        hg₁ : Membership.mem (atlas H M') (chartAt H y)
        hg₂ : Membership.mem (chartAt H y).source y
        s : Set H := Inter.inter (c.symm.trans f₁).source (Set.preimage (↑(c.symm.tran …
        open_s : IsOpen s
        this✝ : Membership.mem s x
        F₁ : PartialHomeomorph H H := (c.symm.trans (f₁.trans g)).trans (g.symm.trans  …
        A : Membership.mem G F₁
        F₂ : PartialHomeomorph H H := (c.symm.trans (f.trans c')).restr s
        this : HasEquiv.Equiv F₁ F₂
        ⊢ Membership.mem G ((c.symm.trans (__src✝.toPartialHomeomorph.trans c')).restr …
      -/
      have : F₂ ∈ G := G.mem_of_eqOnSource A (Setoid.symm this)
      /-
        H : Type u
        H' : Type u_1
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝⁶ : TopologicalSpace H
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : ChartedSpace H M
        inst✝³ : TopologicalSpace M'
        inst✝² : TopologicalSpace M''
        G : StructureGroupoid H
        inst✝¹ : ChartedSpace H M'
        inst✝ : ChartedSpace H M''
        e : Structomorph G M M'
        e' : Structomorph G M' M''
        c : PartialHomeomorph M H
        c' : PartialHomeomorph M'' H
        hc : Membership.mem (atlas H M) c
        hc' : Membership.mem (atlas H M'') c'
        x : H
        hx : Membership.mem (c.symm.trans (__src✝.toPartialHomeomorph.trans c')).sourc …
        f₁ : PartialHomeomorph M M' := e.toPartialHomeomorph
        f₂ : PartialHomeomorph M' M'' := e'.toPartialHomeomorph
        f : PartialHomeomorph M M'' := (e.trans e'.toHomeomorph).toPartialHomeomorph
        feq : Eq f (f₁.trans f₂)
        y : M' := ↑(c.symm.trans f₁) x
        g : PartialHomeomorph M' H := chartAt H y
        hg₁ : Membership.mem (atlas H M') (chartAt H y)
        hg₂ : Membership.mem (chartAt H y).source y
        s : Set H := Inter.inter (c.symm.trans f₁).source (Set.preimage (↑(c.symm.tran …
        open_s : IsOpen s
        this✝¹ : Membership.mem s x
        F₁ : PartialHomeomorph H H := (c.symm.trans (f₁.trans g)).trans (g.symm.trans  …
        A : Membership.mem G F₁
        F₂ : PartialHomeomorph H H := (c.symm.trans (f.trans c')).restr s
        this✝ : HasEquiv.Equiv F₁ F₂
        this : Membership.mem G F₂
        ⊢ Membership.mem G ((c.symm.trans (__src✝.toPartialHomeomorph.trans c')).restr …
      -/
      exact this }
      /-
        🎉 no goals
      -/


/-- Restricting a chart to its source `s ⊆ M` yields a chart in the maximal atlas of `s`. -/
theorem StructureGroupoid.restriction_mem_maximalAtlas_subtype
    {e : PartialHomeomorph M H} (he : e ∈ atlas H M)
    (hs : Nonempty e.source) [HasGroupoid M G] [ClosedUnderRestriction G] :
    let s := { carrier := e.source, is_open' := e.open_source : Opens M }
    let t := { carrier := e.target, is_open' := e.open_target : Opens H }
    ∀ c' ∈ atlas H t, e.toHomeomorphSourceTarget.toPartialHomeomorph ≫ₕ c' ∈ G.maximalAtlas s := by
  /-
    H : Type u
    M : Type u_2
    inst✝⁴ : TopologicalSpace H
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    G : StructureGroupoid H
    e : PartialHomeomorph M H
    he : Membership.mem (atlas H M) e
    hs : Nonempty ↑e.source
    inst✝¹ : HasGroupoid M G
    inst✝ : ClosedUnderRestriction G
    ⊢ let s := { carrier := e.source, is_open' := ⋯ };
      let t := { carrier := e.target, is_open' := ⋯ };
      ∀ (c' : PartialHomeomorph (Subtype fun x => Membership.mem t x) H), Membersh …
  -/
  intro s t c' hc'
  /-
    H : Type u
    M : Type u_2
    inst✝⁴ : TopologicalSpace H
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    G : StructureGroupoid H
    e : PartialHomeomorph M H
    he : Membership.mem (atlas H M) e
    hs : Nonempty ↑e.source
    inst✝¹ : HasGroupoid M G
    inst✝ : ClosedUnderRestriction G
    s : TopologicalSpace.Opens M := { carrier := e.source, is_open' := ⋯ }
    t : TopologicalSpace.Opens H := { carrier := e.target, is_open' := ⋯ }
    c' : PartialHomeomorph (Subtype fun x => Membership.mem t x) H
    hc' : Membership.mem (atlas H (Subtype fun x => Membership.mem t x)) c'
    ⊢ Membership.mem (StructureGroupoid.maximalAtlas (Subtype fun x => Membership. …
  -/
  have : Nonempty t := nonempty_coe_sort.mpr (e.mapsTo.nonempty (nonempty_coe_sort.mp hs))
  /-
    H : Type u
    M : Type u_2
    inst✝⁴ : TopologicalSpace H
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    G : StructureGroupoid H
    e : PartialHomeomorph M H
    he : Membership.mem (atlas H M) e
    hs : Nonempty ↑e.source
    inst✝¹ : HasGroupoid M G
    inst✝ : ClosedUnderRestriction G
    s : TopologicalSpace.Opens M := { carrier := e.source, is_open' := ⋯ }
    t : TopologicalSpace.Opens H := { carrier := e.target, is_open' := ⋯ }
    c' : PartialHomeomorph (Subtype fun x => Membership.mem t x) H
    hc' : Membership.mem (atlas H (Subtype fun x => Membership.mem t x)) c'
    this : Nonempty (Subtype fun x => Membership.mem t x)
    ⊢ Membership.mem (StructureGroupoid.maximalAtlas (Subtype fun x => Membership. …
  -/
  obtain ⟨x, hc'⟩ := Opens.chart_eq this hc'
  -- As H has only one chart, `chartAt H x` is the identity: i.e., `c'` is the inclusion.
  /-
    case intro
    H : Type u
    M : Type u_2
    inst✝⁴ : TopologicalSpace H
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    G : StructureGroupoid H
    e : PartialHomeomorph M H
    he : Membership.mem (atlas H M) e
    hs : Nonempty ↑e.source
    inst✝¹ : HasGroupoid M G
    inst✝ : ClosedUnderRestriction G
    s : TopologicalSpace.Opens M := { carrier := e.source, is_open' := ⋯ }
    t : TopologicalSpace.Opens H := { carrier := e.target, is_open' := ⋯ }
    c' : PartialHomeomorph (Subtype fun x => Membership.mem t x) H
    hc'✝ : Membership.mem (atlas H (Subtype fun x => Membership.mem t x)) c'
    this : Nonempty (Subtype fun x => Membership.mem t x)
    x : Subtype fun x => Membership.mem t x
    hc' : Eq c' ((chartAt H ↑x).subtypeRestr this)
    ⊢ Membership.mem (StructureGroupoid.maximalAtlas (Subtype fun x => Membership. …
  -/
  rw [hc', (chartAt_self_eq)]
  -- Our expression equals this chart, at least on its source.
  /-
    case intro
    H : Type u
    M : Type u_2
    inst✝⁴ : TopologicalSpace H
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    G : StructureGroupoid H
    e : PartialHomeomorph M H
    he : Membership.mem (atlas H M) e
    hs : Nonempty ↑e.source
    inst✝¹ : HasGroupoid M G
    inst✝ : ClosedUnderRestriction G
    s : TopologicalSpace.Opens M := { carrier := e.source, is_open' := ⋯ }
    t : TopologicalSpace.Opens H := { carrier := e.target, is_open' := ⋯ }
    c' : PartialHomeomorph (Subtype fun x => Membership.mem t x) H
    hc'✝ : Membership.mem (atlas H (Subtype fun x => Membership.mem t x)) c'
    this : Nonempty (Subtype fun x => Membership.mem t x)
    x : Subtype fun x => Membership.mem t x
    hc' : Eq c' ((chartAt H ↑x).subtypeRestr this)
    ⊢ Membership.mem (StructureGroupoid.maximalAtlas (Subtype fun x => Membership. …
  -/
  rw [PartialHomeomorph.subtypeRestr_def, PartialHomeomorph.trans_refl]
  /-
    case intro
    H : Type u
    M : Type u_2
    inst✝⁴ : TopologicalSpace H
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    G : StructureGroupoid H
    e : PartialHomeomorph M H
    he : Membership.mem (atlas H M) e
    hs : Nonempty ↑e.source
    inst✝¹ : HasGroupoid M G
    inst✝ : ClosedUnderRestriction G
    s : TopologicalSpace.Opens M := { carrier := e.source, is_open' := ⋯ }
    t : TopologicalSpace.Opens H := { carrier := e.target, is_open' := ⋯ }
    c' : PartialHomeomorph (Subtype fun x => Membership.mem t x) H
    hc'✝ : Membership.mem (atlas H (Subtype fun x => Membership.mem t x)) c'
    this : Nonempty (Subtype fun x => Membership.mem t x)
    x : Subtype fun x => Membership.mem t x
    hc' : Eq c' ((chartAt H ↑x).subtypeRestr this)
    ⊢ Membership.mem (StructureGroupoid.maximalAtlas (Subtype fun x => Membership. …
  -/
  let goal := e.toHomeomorphSourceTarget.toPartialHomeomorph ≫ₕ (t.partialHomeomorphSubtypeCoe this)
  have : goal ≈ e.subtypeRestr (s := s) hs :=
    (goal.eqOnSource_iff (e.subtypeRestr (s := s) hs)).mpr
      ⟨by
        simp only [trans_toPartialEquiv, PartialEquiv.trans_source,
          Homeomorph.toPartialHomeomorph_source, toFun_eq_coe, Homeomorph.toPartialHomeomorph_apply,
          Opens.partialHomeomorphSubtypeCoe_source, preimage_univ, inter_self, subtypeRestr_source,
          goal, s]
        exact Subtype.coe_preimage_self _ |>.symm, by intro _ _; rfl⟩
  /-
    case intro
    H : Type u
    M : Type u_2
    inst✝⁴ : TopologicalSpace H
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    G : StructureGroupoid H
    e : PartialHomeomorph M H
    he : Membership.mem (atlas H M) e
    hs : Nonempty ↑e.source
    inst✝¹ : HasGroupoid M G
    inst✝ : ClosedUnderRestriction G
    s : TopologicalSpace.Opens M := { carrier := e.source, is_open' := ⋯ }
    t : TopologicalSpace.Opens H := { carrier := e.target, is_open' := ⋯ }
    c' : PartialHomeomorph (Subtype fun x => Membership.mem t x) H
    hc'✝ : Membership.mem (atlas H (Subtype fun x => Membership.mem t x)) c'
    this✝ : Nonempty (Subtype fun x => Membership.mem t x)
    x : Subtype fun x => Membership.mem t x
    hc' : Eq c' ((chartAt H ↑x).subtypeRestr this✝)
    goal : PartialHomeomorph (↑e.source) H := e.toHomeomorphSourceTarget.toPartial …
    this : HasEquiv.Equiv goal (e.subtypeRestr hs)
    ⊢ Membership.mem (StructureGroupoid.maximalAtlas (Subtype fun x => Membership. …
  -/
  exact G.mem_maximalAtlas_of_eqOnSource (M := s) this (G.restriction_in_maximalAtlas he hs)
  /-
    🎉 no goals
  -/


/-- Each chart of a charted space is a structomorphism between its source and target. -/
def PartialHomeomorph.toStructomorph {e : PartialHomeomorph M H} (he : e ∈ atlas H M)
    [HasGroupoid M G] [ClosedUnderRestriction G] :
    let s : Opens M := { carrier := e.source, is_open' := e.open_source }
    let t : Opens H := { carrier := e.target, is_open' := e.open_target }
    Structomorph G s t := by
  /-
    H : Type u
    H' : Type u_1
    M : Type u_2
    M' : Type u_3
    M'' : Type u_4
    inst✝⁸ : TopologicalSpace H
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : ChartedSpace H M
    inst✝⁵ : TopologicalSpace M'
    inst✝⁴ : TopologicalSpace M''
    G : StructureGroupoid H
    inst✝³ : ChartedSpace H M'
    inst✝² : ChartedSpace H M''
    e : PartialHomeomorph M H
    he : Membership.mem (atlas H M) e
    inst✝¹ : HasGroupoid M G
    inst✝ : ClosedUnderRestriction G
    ⊢ let s := { carrier := e.source, is_open' := ⋯ };
      let t := { carrier := e.target, is_open' := ⋯ };
      Structomorph G (Subtype fun x => Membership.mem s x) (Subtype fun x => Membe …
  -/
  intro s t
  /-
    H : Type u
    H' : Type u_1
    M : Type u_2
    M' : Type u_3
    M'' : Type u_4
    inst✝⁸ : TopologicalSpace H
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : ChartedSpace H M
    inst✝⁵ : TopologicalSpace M'
    inst✝⁴ : TopologicalSpace M''
    G : StructureGroupoid H
    inst✝³ : ChartedSpace H M'
    inst✝² : ChartedSpace H M''
    e : PartialHomeomorph M H
    he : Membership.mem (atlas H M) e
    inst✝¹ : HasGroupoid M G
    inst✝ : ClosedUnderRestriction G
    s : TopologicalSpace.Opens M := { carrier := e.source, is_open' := ⋯ }
    t : TopologicalSpace.Opens H := { carrier := e.target, is_open' := ⋯ }
    ⊢ Structomorph G (Subtype fun x => Membership.mem s x) (Subtype fun x => Membe …
  -/
  by_cases h : Nonempty e.source
  · exact { e.toHomeomorphSourceTarget with
      mem_groupoid :=
        -- The atlas of H on itself has only one chart, hence c' is the inclusion.
        -- Then, compatibility of `G` *almost* yields our claim --- except that `e` is a chart
        -- on `M` and `c` is one on `s`: we need to show that restricting `e` to `s` and composing
        -- with `c'` yields a chart in the maximal atlas of `s`.
        fun c c' hc hc' ↦ G.compatible_of_mem_maximalAtlas (G.subset_maximalAtlas hc)
          (G.restriction_mem_maximalAtlas_subtype he h c' hc') }
    /-
      case neg
      H : Type u
      H' : Type u_1
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝⁸ : TopologicalSpace H
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : ChartedSpace H M
      inst✝⁵ : TopologicalSpace M'
      inst✝⁴ : TopologicalSpace M''
      G : StructureGroupoid H
      inst✝³ : ChartedSpace H M'
      inst✝² : ChartedSpace H M''
      e : PartialHomeomorph M H
      he : Membership.mem (atlas H M) e
      inst✝¹ : HasGroupoid M G
      inst✝ : ClosedUnderRestriction G
      s : TopologicalSpace.Opens M := { carrier := e.source, is_open' := ⋯ }
      t : TopologicalSpace.Opens H := { carrier := e.target, is_open' := ⋯ }
      h : Not (Nonempty ↑e.source)
      ⊢ Structomorph G (Subtype fun x => Membership.mem s x) (Subtype fun x => Membe …
    -/
  · have : IsEmpty s := not_nonempty_iff.mp h
    have : IsEmpty t := isEmpty_coe_sort.mpr
      (by convert e.image_source_eq_target ▸ image_eq_empty.mpr (isEmpty_coe_sort.mp this))
    exact { Homeomorph.empty with
      -- `c'` cannot exist: it would be the restriction of `chartAt H x` at some `x ∈ t`.
      mem_groupoid := fun _ c' _ ⟨_, ⟨x, _⟩, _⟩ ↦ (this.false x).elim }


