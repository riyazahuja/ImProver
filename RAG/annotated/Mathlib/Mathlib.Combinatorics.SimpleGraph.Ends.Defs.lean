/-- The components outside a given set of vertices `K` -/
abbrev ComponentCompl :=
  (G.induce Kᶜ).ConnectedComponent


/-- The connected component of `v` in `G.induce Kᶜ`. -/
abbrev componentComplMk (G : SimpleGraph V) {v : V} (vK : v ∉ K) : G.ComponentCompl K :=
  connectedComponentMk (G.induce Kᶜ) ⟨v, vK⟩


/-- The set of vertices of `G` making up the connected component `C` -/
def ComponentCompl.supp (C : G.ComponentCompl K) : Set V :=
  { v : V | ∃ h : v ∉ K, G.componentComplMk h = C }


@[ext]
theorem ComponentCompl.supp_injective :
    Function.Injective (ComponentCompl.supp : G.ComponentCompl K → Set V) := by
  /-
    V : Type u
    G : SimpleGraph V
    K : Set V
    ⊢ Function.Injective SimpleGraph.ComponentCompl.supp
  -/
  refine ConnectedComponent.ind₂ ?_
  /-
    V : Type u
    G : SimpleGraph V
    K : Set V
    ⊢ ∀ (v w : ↑(HasCompl.compl K)), Eq (SimpleGraph.ComponentCompl.supp ((SimpleG …
  -/
  rintro ⟨v, hv⟩ ⟨w, hw⟩ h
  /-
    case mk.mk
    V : Type u
    G : SimpleGraph V
    K : Set V
    v : V
    hv : Membership.mem (HasCompl.compl K) v
    w : V
    hw : Membership.mem (HasCompl.compl K) w
    h : Eq (SimpleGraph.ComponentCompl.supp ((SimpleGraph.induce (HasCompl.compl K …
    ⊢ Eq ((SimpleGraph.induce (HasCompl.compl K) G).connectedComponentMk ⟨v, hv⟩)  …
  -/
  simp only [Set.ext_iff, ConnectedComponent.eq, Set.mem_setOf_eq, ComponentCompl.supp] at h ⊢
  /-
    case mk.mk
    V : Type u
    G : SimpleGraph V
    K : Set V
    v : V
    hv : Membership.mem (HasCompl.compl K) v
    w : V
    hw : Membership.mem (HasCompl.compl K) w
    h : ∀ (x : V), Iff (Exists fun h => (SimpleGraph.induce (HasCompl.compl K) G). …
    ⊢ (SimpleGraph.induce (HasCompl.compl K) G).Reachable ⟨v, hv⟩ ⟨w, hw⟩
  -/
  exact ((h v).mp ⟨hv, Reachable.refl _⟩).choose_spec
  /-
    🎉 no goals
  -/


theorem ComponentCompl.supp_inj {C D : G.ComponentCompl K} : C.supp = D.supp ↔ C = D :=
  ComponentCompl.supp_injective.eq_iff


instance ComponentCompl.setLike : SetLike (G.ComponentCompl K) V where
  coe := ComponentCompl.supp
  coe_injective' _ _ := ComponentCompl.supp_inj.mp


@[simp]
theorem ComponentCompl.mem_supp_iff {v : V} {C : ComponentCompl G K} :
    v ∈ C ↔ ∃ vK : v ∉ K, G.componentComplMk vK = C :=
  Iff.rfl


theorem componentComplMk_mem (G : SimpleGraph V) {v : V} (vK : v ∉ K) : v ∈ G.componentComplMk vK :=
  ⟨vK, rfl⟩


theorem componentComplMk_eq_of_adj (G : SimpleGraph V) {v w : V} (vK : v ∉ K) (wK : w ∉ K)
    (a : G.Adj v w) : G.componentComplMk vK = G.componentComplMk wK := by
  /-
    V : Type u
    K : Set V
    G : SimpleGraph V
    v w : V
    vK : Not (Membership.mem K v)
    wK : Not (Membership.mem K w)
    a : G.Adj v w
    ⊢ Eq (G.componentComplMk vK) (G.componentComplMk wK)
  -/
  rw [ConnectedComponent.eq]
  /-
    V : Type u
    K : Set V
    G : SimpleGraph V
    v w : V
    vK : Not (Membership.mem K v)
    wK : Not (Membership.mem K w)
    a : G.Adj v w
    ⊢ (SimpleGraph.induce (HasCompl.compl K) G).Reachable ⟨v, vK⟩ ⟨w, wK⟩
  -/
  apply Adj.reachable
  /-
    case h
    V : Type u
    K : Set V
    G : SimpleGraph V
    v w : V
    vK : Not (Membership.mem K v)
    wK : Not (Membership.mem K w)
    a : G.Adj v w
    ⊢ (SimpleGraph.induce (HasCompl.compl K) G).Adj ⟨v, vK⟩ ⟨w, wK⟩
  -/
  exact a
  /-
    🎉 no goals
  -/


/-- In an infinite graph, the set of components out of a finite set is nonempty. -/
instance componentCompl_nonempty_of_infinite (G : SimpleGraph V) [Infinite V] (K : Finset V) :
    Nonempty (G.ComponentCompl K) :=
  let ⟨_, kK⟩ := K.finite_toSet.infinite_compl.nonempty
  ⟨componentComplMk _ kK⟩


/-- A `ComponentCompl` specialization of `Quot.lift`, where soundness has to be proved only
for adjacent vertices.
-/
protected def lift {β : Sort*} (f : ∀ ⦃v⦄ (_ : v ∉ K), β)
    (h : ∀ ⦃v w⦄ (hv : v ∉ K) (hw : w ∉ K), G.Adj v w → f hv = f hw) : G.ComponentCompl K → β :=
  ConnectedComponent.lift (fun vv => f vv.prop) fun v w p => by
    /-
      V : Type u
      G : SimpleGraph V
      K L M : Set V
      β : Sort u_1
      f : ⦃v : V⦄ → Not (Membership.mem K v) → β
      h : ∀ ⦃v w : V⦄ (hv : Not (Membership.mem K v)) (hw : Not (Membership.mem K w) …
      v w : ↑(HasCompl.compl K)
      p : (SimpleGraph.induce (HasCompl.compl K) G).Walk v w
      ⊢ p.IsPath → Eq ((fun vv => f ⋯) v) ((fun vv => f ⋯) w)
    -/
    induction' p with _ u v w a q ih
      /-
        case nil
        V : Type u
        G : SimpleGraph V
        K L M : Set V
        β : Sort u_1
        f : ⦃v : V⦄ → Not (Membership.mem K v) → β
        h : ∀ ⦃v w : V⦄ (hv : Not (Membership.mem K v)) (hw : Not (Membership.mem K w) …
        v w u✝ : ↑(HasCompl.compl K)
        ⊢ SimpleGraph.Walk.nil.IsPath → Eq ((fun vv => f ⋯) u✝) ((fun vv => f ⋯) u✝)
      -/
    · rintro _
      /-
        case nil
        V : Type u
        G : SimpleGraph V
        K L M : Set V
        β : Sort u_1
        f : ⦃v : V⦄ → Not (Membership.mem K v) → β
        h : ∀ ⦃v w : V⦄ (hv : Not (Membership.mem K v)) (hw : Not (Membership.mem K w) …
        v w u✝ : ↑(HasCompl.compl K)
        a✝ : SimpleGraph.Walk.nil.IsPath
        ⊢ Eq ((fun vv => f ⋯) u✝) ((fun vv => f ⋯) u✝)
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case cons
        V : Type u
        G : SimpleGraph V
        K L M : Set V
        β : Sort u_1
        f : ⦃v : V⦄ → Not (Membership.mem K v) → β
        h : ∀ ⦃v w : V⦄ (hv : Not (Membership.mem K v)) (hw : Not (Membership.mem K w) …
        v✝ w✝ u v w : ↑(HasCompl.compl K)
        a : (SimpleGraph.induce (HasCompl.compl K) G).Adj u v
        q : (SimpleGraph.induce (HasCompl.compl K) G).Walk v w
        ih : q.IsPath → Eq ((fun vv => f ⋯) v) ((fun vv => f ⋯) w)
        ⊢ (SimpleGraph.Walk.cons a q).IsPath → Eq ((fun vv => f ⋯) u) ((fun vv => f ⋯) …
      -/
    · rintro h'
      /-
        case cons
        V : Type u
        G : SimpleGraph V
        K L M : Set V
        β : Sort u_1
        f : ⦃v : V⦄ → Not (Membership.mem K v) → β
        h : ∀ ⦃v w : V⦄ (hv : Not (Membership.mem K v)) (hw : Not (Membership.mem K w) …
        v✝ w✝ u v w : ↑(HasCompl.compl K)
        a : (SimpleGraph.induce (HasCompl.compl K) G).Adj u v
        q : (SimpleGraph.induce (HasCompl.compl K) G).Walk v w
        ih : q.IsPath → Eq ((fun vv => f ⋯) v) ((fun vv => f ⋯) w)
        h' : (SimpleGraph.Walk.cons a q).IsPath
        ⊢ Eq ((fun vv => f ⋯) u) ((fun vv => f ⋯) w)
      -/
      exact (h u.prop v.prop a).trans (ih h'.of_cons)
      /-
        🎉 no goals
      -/


@[elab_as_elim] -- Porting note: added
protected theorem ind {β : G.ComponentCompl K → Prop}
    (f : ∀ ⦃v⦄ (hv : v ∉ K), β (G.componentComplMk hv)) : ∀ C : G.ComponentCompl K, β C := by
  /-
    V : Type u
    G : SimpleGraph V
    K : Set V
    β : G.ComponentCompl K → Prop
    f : ∀ ⦃v : V⦄ (hv : Not (Membership.mem K v)), β (G.componentComplMk hv)
    ⊢ ∀ (C : G.ComponentCompl K), β C
  -/
  apply ConnectedComponent.ind
  /-
    case h
    V : Type u
    G : SimpleGraph V
    K : Set V
    β : G.ComponentCompl K → Prop
    f : ∀ ⦃v : V⦄ (hv : Not (Membership.mem K v)), β (G.componentComplMk hv)
    ⊢ ∀ (v : ↑(HasCompl.compl K)), β ((SimpleGraph.induce (HasCompl.compl K) G).co …
  -/
  exact fun ⟨v, vnK⟩ => f vnK
  /-
    🎉 no goals
  -/


/-- The induced graph on the vertices `C`. -/
protected abbrev coeGraph (C : ComponentCompl G K) : SimpleGraph C :=
  G.induce (C : Set V)


theorem coe_inj {C D : G.ComponentCompl K} : (C : Set V) = (D : Set V) ↔ C = D :=
  SetLike.coe_set_eq


@[simp]
protected theorem nonempty (C : G.ComponentCompl K) : (C : Set V).Nonempty :=
  C.ind fun v vnK => ⟨v, vnK, rfl⟩


protected theorem exists_eq_mk (C : G.ComponentCompl K) :
    ∃ (v : _) (h : v ∉ K), G.componentComplMk h = C :=
  C.nonempty


protected theorem disjoint_right (C : G.ComponentCompl K) : Disjoint K C := by
  /-
    V : Type u
    G : SimpleGraph V
    K : Set V
    C : G.ComponentCompl K
    ⊢ Disjoint K ↑C
  -/
  rw [Set.disjoint_iff]
  /-
    V : Type u
    G : SimpleGraph V
    K : Set V
    C : G.ComponentCompl K
    ⊢ HasSubset.Subset (Inter.inter K ↑C) EmptyCollection.emptyCollection
  -/
  exact fun v ⟨vK, vC⟩ => vC.choose vK
  /-
    🎉 no goals
  -/


theorem not_mem_of_mem {C : G.ComponentCompl K} {c : V} (cC : c ∈ C) : c ∉ K := fun cK =>
  Set.disjoint_iff.mp C.disjoint_right ⟨cK, cC⟩


protected theorem pairwise_disjoint :
    Pairwise fun C D : G.ComponentCompl K => Disjoint (C : Set V) (D : Set V) := by
  /-
    V : Type u
    G : SimpleGraph V
    K : Set V
    ⊢ Pairwise fun C D => Disjoint ↑C ↑D
  -/
  rintro C D ne
  /-
    V : Type u
    G : SimpleGraph V
    K : Set V
    C D : G.ComponentCompl K
    ne : Ne C D
    ⊢ Disjoint ↑C ↑D
  -/
  rw [Set.disjoint_iff]
  /-
    V : Type u
    G : SimpleGraph V
    K : Set V
    C D : G.ComponentCompl K
    ne : Ne C D
    ⊢ HasSubset.Subset (Inter.inter ↑C ↑D) EmptyCollection.emptyCollection
  -/
  exact fun u ⟨uC, uD⟩ => ne (uC.choose_spec.symm.trans uD.choose_spec)
  /-
    🎉 no goals
  -/


/-- Any vertex adjacent to a vertex of `C` and not lying in `K` must lie in `C`.
-/
theorem mem_of_adj : ∀ {C : G.ComponentCompl K} (c d : V), c ∈ C → d ∉ K → G.Adj c d → d ∈ C :=
  fun {C} c d ⟨cnK, h⟩ dnK cd =>
  ⟨dnK, by
    /-
      V : Type u
      G : SimpleGraph V
      K : Set V
      C : G.ComponentCompl K
      c d : V
      x✝ : Membership.mem C c
      dnK : Not (Membership.mem K d)
      cd : G.Adj c d
      cnK : Not (Membership.mem K c)
      h : Eq (G.componentComplMk cnK) C
      ⊢ Eq (G.componentComplMk dnK) C
    -/
    rw [← h, ConnectedComponent.eq]
    /-
      V : Type u
      G : SimpleGraph V
      K : Set V
      C : G.ComponentCompl K
      c d : V
      x✝ : Membership.mem C c
      dnK : Not (Membership.mem K d)
      cd : G.Adj c d
      cnK : Not (Membership.mem K c)
      h : Eq (G.componentComplMk cnK) C
      ⊢ (SimpleGraph.induce (HasCompl.compl K) G).Reachable ⟨d, dnK⟩ ⟨c, cnK⟩
    -/
    exact Adj.reachable cd.symm⟩
    /-
      🎉 no goals
    -/


/--
Assuming `G` is preconnected and `K` not empty, given any connected component `C` outside of `K`,
there exists a vertex `k ∈ K` adjacent to a vertex `v ∈ C`.
-/
theorem exists_adj_boundary_pair (Gc : G.Preconnected) (hK : K.Nonempty) :
    ∀ C : G.ComponentCompl K, ∃ ck : V × V, ck.1 ∈ C ∧ ck.2 ∈ K ∧ G.Adj ck.1 ck.2 := by
  /-
    V : Type u
    G : SimpleGraph V
    K : Set V
    Gc : G.Preconnected
    hK : K.Nonempty
    ⊢ ∀ (C : G.ComponentCompl K), Exists fun ck => And (Membership.mem C ck.1) (An …
  -/
  refine ComponentCompl.ind fun v vnK => ?_
  /-
    V : Type u
    G : SimpleGraph V
    K : Set V
    Gc : G.Preconnected
    hK : K.Nonempty
    v : V
    vnK : Not (Membership.mem K v)
    ⊢ Exists fun ck => And (Membership.mem (G.componentComplMk vnK) ck.1) (And (Me …
  -/
  let C : G.ComponentCompl K := G.componentComplMk vnK
  /-
    V : Type u
    G : SimpleGraph V
    K : Set V
    Gc : G.Preconnected
    hK : K.Nonempty
    v : V
    vnK : Not (Membership.mem K v)
    C : G.ComponentCompl K := G.componentComplMk vnK
    ⊢ Exists fun ck => And (Membership.mem (G.componentComplMk vnK) ck.1) (And (Me …
  -/
  let dis := Set.disjoint_iff.mp C.disjoint_right
  /-
    V : Type u
    G : SimpleGraph V
    K : Set V
    Gc : G.Preconnected
    hK : K.Nonempty
    v : V
    vnK : Not (Membership.mem K v)
    C : G.ComponentCompl K := G.componentComplMk vnK
    dis : HasSubset.Subset (Inter.inter K ↑C) EmptyCollection.emptyCollection := S …
    ⊢ Exists fun ck => And (Membership.mem (G.componentComplMk vnK) ck.1) (And (Me …
  -/
  by_contra! h
  /-
    V : Type u
    G : SimpleGraph V
    K : Set V
    Gc : G.Preconnected
    hK : K.Nonempty
    v : V
    vnK : Not (Membership.mem K v)
    C : G.ComponentCompl K := G.componentComplMk vnK
    dis : HasSubset.Subset (Inter.inter K ↑C) EmptyCollection.emptyCollection := S …
    h : ∀ (ck : Prod V V), Membership.mem (G.componentComplMk vnK) ck.1 → Membersh …
    ⊢ False
  -/
  suffices Set.univ = (C : Set V) by exact dis ⟨hK.choose_spec, this ▸ Set.mem_univ hK.some⟩
  /-
    V : Type u
    G : SimpleGraph V
    K : Set V
    Gc : G.Preconnected
    hK : K.Nonempty
    v : V
    vnK : Not (Membership.mem K v)
    C : G.ComponentCompl K := G.componentComplMk vnK
    dis : HasSubset.Subset (Inter.inter K ↑C) EmptyCollection.emptyCollection := S …
    h : ∀ (ck : Prod V V), Membership.mem (G.componentComplMk vnK) ck.1 → Membersh …
    ⊢ Eq Set.univ ↑C
  -/
  symm
  /-
    V : Type u
    G : SimpleGraph V
    K : Set V
    Gc : G.Preconnected
    hK : K.Nonempty
    v : V
    vnK : Not (Membership.mem K v)
    C : G.ComponentCompl K := G.componentComplMk vnK
    dis : HasSubset.Subset (Inter.inter K ↑C) EmptyCollection.emptyCollection := S …
    h : ∀ (ck : Prod V V), Membership.mem (G.componentComplMk vnK) ck.1 → Membersh …
    ⊢ Eq (↑C) Set.univ
  -/
  rw [Set.eq_univ_iff_forall]
  /-
    V : Type u
    G : SimpleGraph V
    K : Set V
    Gc : G.Preconnected
    hK : K.Nonempty
    v : V
    vnK : Not (Membership.mem K v)
    C : G.ComponentCompl K := G.componentComplMk vnK
    dis : HasSubset.Subset (Inter.inter K ↑C) EmptyCollection.emptyCollection := S …
    h : ∀ (ck : Prod V V), Membership.mem (G.componentComplMk vnK) ck.1 → Membersh …
    ⊢ ∀ (x : V), Membership.mem (↑C) x
  -/
  rintro u
  /-
    V : Type u
    G : SimpleGraph V
    K : Set V
    Gc : G.Preconnected
    hK : K.Nonempty
    v : V
    vnK : Not (Membership.mem K v)
    C : G.ComponentCompl K := G.componentComplMk vnK
    dis : HasSubset.Subset (Inter.inter K ↑C) EmptyCollection.emptyCollection := S …
    h : ∀ (ck : Prod V V), Membership.mem (G.componentComplMk vnK) ck.1 → Membersh …
    u : V
    ⊢ Membership.mem (↑C) u
  -/
  by_contra unC
  /-
    V : Type u
    G : SimpleGraph V
    K : Set V
    Gc : G.Preconnected
    hK : K.Nonempty
    v : V
    vnK : Not (Membership.mem K v)
    C : G.ComponentCompl K := G.componentComplMk vnK
    dis : HasSubset.Subset (Inter.inter K ↑C) EmptyCollection.emptyCollection := S …
    h : ∀ (ck : Prod V V), Membership.mem (G.componentComplMk vnK) ck.1 → Membersh …
    u : V
    unC : Not (Membership.mem (↑C) u)
    ⊢ False
  -/
  obtain ⟨p⟩ := Gc v u
  obtain ⟨⟨⟨x, y⟩, xy⟩, -, xC, ynC⟩ :=
    p.exists_boundary_dart (C : Set V) (G.componentComplMk_mem vnK) unC
  /-
    case intro.intro.mk.mk.intro.intro
    V : Type u
    G : SimpleGraph V
    K : Set V
    Gc : G.Preconnected
    hK : K.Nonempty
    v : V
    vnK : Not (Membership.mem K v)
    C : G.ComponentCompl K := G.componentComplMk vnK
    dis : HasSubset.Subset (Inter.inter K ↑C) EmptyCollection.emptyCollection := S …
    h : ∀ (ck : Prod V V), Membership.mem (G.componentComplMk vnK) ck.1 → Membersh …
    u : V
    unC : Not (Membership.mem (↑C) u)
    p : G.Walk v u
    x y : V
    xy : G.Adj { fst := x, snd := y }.1 { fst := x, snd := y }.2
    xC : Membership.mem ↑C { fst := x, snd := y, adj := xy }.toProd.1
    ynC : Not (Membership.mem ↑C { fst := x, snd := y, adj := xy }.toProd.2)
    ⊢ False
  -/
  exact ynC (mem_of_adj x y xC (fun yK : y ∈ K => h ⟨x, y⟩ xC yK xy) xy)
  /-
    🎉 no goals
  -/


/--
If `K ⊆ L`, the components outside of `L` are all contained in a single component outside of `K`.
-/
abbrev hom (h : K ⊆ L) (C : G.ComponentCompl L) : G.ComponentCompl K :=
  C.map <| induceHom Hom.id <| Set.compl_subset_compl.2 h


theorem subset_hom (C : G.ComponentCompl L) (h : K ⊆ L) : (C : Set V) ⊆ (C.hom h : Set V) := by
  /-
    V : Type u
    G : SimpleGraph V
    K L : Set V
    C : G.ComponentCompl L
    h : HasSubset.Subset K L
    ⊢ HasSubset.Subset ↑C ↑(SimpleGraph.ComponentCompl.hom h C)
  -/
  rintro c ⟨cL, rfl⟩
  /-
    case intro
    V : Type u
    G : SimpleGraph V
    K L : Set V
    h : HasSubset.Subset K L
    c : V
    cL : Not (Membership.mem L c)
    ⊢ Membership.mem (↑(SimpleGraph.ComponentCompl.hom h (G.componentComplMk cL))) c
  -/
  exact ⟨fun h' => cL (h h'), rfl⟩
  /-
    🎉 no goals
  -/


theorem _root_.SimpleGraph.componentComplMk_mem_hom
    (G : SimpleGraph V) {v : V} (vK : v ∉ K) (h : L ⊆ K) :
    v ∈ (G.componentComplMk vK).hom h :=
  subset_hom (G.componentComplMk vK) h (G.componentComplMk_mem vK)


theorem hom_eq_iff_le (C : G.ComponentCompl L) (h : K ⊆ L) (D : G.ComponentCompl K) :
    C.hom h = D ↔ (C : Set V) ⊆ (D : Set V) :=
  ⟨fun h' => h' ▸ C.subset_hom h, C.ind fun _ vnL vD => (vD ⟨vnL, rfl⟩).choose_spec⟩


theorem hom_eq_iff_not_disjoint (C : G.ComponentCompl L) (h : K ⊆ L) (D : G.ComponentCompl K) :
    C.hom h = D ↔ ¬Disjoint (C : Set V) (D : Set V) := by
  /-
    V : Type u
    G : SimpleGraph V
    K L : Set V
    C : G.ComponentCompl L
    h : HasSubset.Subset K L
    D : G.ComponentCompl K
    ⊢ Iff (Eq (SimpleGraph.ComponentCompl.hom h C) D) (Not (Disjoint ↑C ↑D))
  -/
  rw [Set.not_disjoint_iff]
  /-
    V : Type u
    G : SimpleGraph V
    K L : Set V
    C : G.ComponentCompl L
    h : HasSubset.Subset K L
    D : G.ComponentCompl K
    ⊢ Iff (Eq (SimpleGraph.ComponentCompl.hom h C) D) (Exists fun x => And (Member …
  -/
  constructor
    /-
      case mp
      V : Type u
      G : SimpleGraph V
      K L : Set V
      C : G.ComponentCompl L
      h : HasSubset.Subset K L
      D : G.ComponentCompl K
      ⊢ Eq (SimpleGraph.ComponentCompl.hom h C) D → Exists fun x => And (Membership. …
    -/
  · rintro rfl
    /-
      case mp
      V : Type u
      G : SimpleGraph V
      K L : Set V
      C : G.ComponentCompl L
      h : HasSubset.Subset K L
      ⊢ Exists fun x => And (Membership.mem (↑C) x) (Membership.mem (↑(SimpleGraph.C …
    -/
    refine C.ind fun x xnL => ?_
    /-
      case mp
      V : Type u
      G : SimpleGraph V
      K L : Set V
      C : G.ComponentCompl L
      h : HasSubset.Subset K L
      x : V
      xnL : Not (Membership.mem L x)
      ⊢ Exists fun x_1 => And (Membership.mem (↑(G.componentComplMk xnL)) x_1) (Memb …
    -/
    exact ⟨x, ⟨xnL, rfl⟩, ⟨fun xK => xnL (h xK), rfl⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      V : Type u
      G : SimpleGraph V
      K L : Set V
      C : G.ComponentCompl L
      h : HasSubset.Subset K L
      D : G.ComponentCompl K
      ⊢ (Exists fun x => And (Membership.mem (↑C) x) (Membership.mem (↑D) x)) → Eq ( …
    -/
  · refine C.ind fun x xnL => ?_
    /-
      case mpr
      V : Type u
      G : SimpleGraph V
      K L : Set V
      C : G.ComponentCompl L
      h : HasSubset.Subset K L
      D : G.ComponentCompl K
      x : V
      xnL : Not (Membership.mem L x)
      ⊢ (Exists fun x_1 => And (Membership.mem (↑(G.componentComplMk xnL)) x_1) (Mem …
    -/
    rintro ⟨x, ⟨_, e₁⟩, _, rfl⟩
    /-
      case mpr.intro.intro.intro.intro
      V : Type u
      G : SimpleGraph V
      K L : Set V
      C : G.ComponentCompl L
      h : HasSubset.Subset K L
      x✝ : V
      xnL : Not (Membership.mem L x✝)
      x : V
      w✝¹ : Not (Membership.mem L x)
      e₁ : Eq (G.componentComplMk w✝¹) (G.componentComplMk xnL)
      w✝ : Not (Membership.mem K x)
      ⊢ Eq (SimpleGraph.ComponentCompl.hom h (G.componentComplMk xnL)) (G.componentC …
    -/
    rw [← e₁]
    /-
      case mpr.intro.intro.intro.intro
      V : Type u
      G : SimpleGraph V
      K L : Set V
      C : G.ComponentCompl L
      h : HasSubset.Subset K L
      x✝ : V
      xnL : Not (Membership.mem L x✝)
      x : V
      w✝¹ : Not (Membership.mem L x)
      e₁ : Eq (G.componentComplMk w✝¹) (G.componentComplMk xnL)
      w✝ : Not (Membership.mem K x)
      ⊢ Eq (SimpleGraph.ComponentCompl.hom h (G.componentComplMk w✝¹)) (G.componentC …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem hom_refl (C : G.ComponentCompl L) : C.hom (subset_refl L) = C := by
  /-
    V : Type u
    G : SimpleGraph V
    L : Set V
    C : G.ComponentCompl L
    ⊢ Eq (SimpleGraph.ComponentCompl.hom ⋯ C) C
  -/
  change C.map _ = C
  /-
    V : Type u
    G : SimpleGraph V
    L : Set V
    C : G.ComponentCompl L
    ⊢ Eq (SimpleGraph.ConnectedComponent.map (SimpleGraph.induceHom SimpleGraph.Ho …
  -/
  rw [induceHom_id G Lᶜ, ConnectedComponent.map_id]
  /-
    🎉 no goals
  -/


theorem hom_trans (C : G.ComponentCompl L) (h : K ⊆ L) (h' : M ⊆ K) :
    C.hom (h'.trans h) = (C.hom h).hom h' := by
  /-
    V : Type u
    G : SimpleGraph V
    K L M : Set V
    C : G.ComponentCompl L
    h : HasSubset.Subset K L
    h' : HasSubset.Subset M K
    ⊢ Eq (SimpleGraph.ComponentCompl.hom ⋯ C) (SimpleGraph.ComponentCompl.hom h' ( …
  -/
  change C.map _ = (C.map _).map _
  /-
    V : Type u
    G : SimpleGraph V
    K L M : Set V
    C : G.ComponentCompl L
    h : HasSubset.Subset K L
    h' : HasSubset.Subset M K
    ⊢ Eq (SimpleGraph.ConnectedComponent.map (SimpleGraph.induceHom SimpleGraph.Ho …
  -/
  rw [ConnectedComponent.map_comp, induceHom_comp]
  /-
    V : Type u
    G : SimpleGraph V
    K L M : Set V
    C : G.ComponentCompl L
    h : HasSubset.Subset K L
    h' : HasSubset.Subset M K
    ⊢ Eq (SimpleGraph.ConnectedComponent.map (SimpleGraph.induceHom SimpleGraph.Ho …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem hom_mk {v : V} (vnL : v ∉ L) (h : K ⊆ L) :
    (G.componentComplMk vnL).hom h = G.componentComplMk (Set.not_mem_subset h vnL) :=
  rfl


theorem hom_infinite (C : G.ComponentCompl L) (h : K ⊆ L) (Cinf : (C : Set V).Infinite) :
    (C.hom h : Set V).Infinite :=
  Set.Infinite.mono (C.subset_hom h) Cinf


theorem infinite_iff_in_all_ranges {K : Finset V} (C : G.ComponentCompl K) :
    C.supp.Infinite ↔ ∀ (L) (h : K ⊆ L), ∃ D : G.ComponentCompl L, D.hom h = C := by
  classical
    constructor
    · rintro Cinf L h
      obtain ⟨v, ⟨vK, rfl⟩, vL⟩ := Set.Infinite.nonempty (Set.Infinite.diff Cinf L.finite_toSet)
      exact ⟨componentComplMk _ vL, rfl⟩
    · rintro h Cfin
      obtain ⟨D, e⟩ := h (K ∪ Cfin.toFinset) Finset.subset_union_left
      obtain ⟨v, vD⟩ := D.nonempty
      let Ddis := D.disjoint_right
      simp_rw [Finset.coe_union, Set.Finite.coe_toFinset, Set.disjoint_union_left,
        Set.disjoint_iff] at Ddis
      exact Ddis.right ⟨(ComponentCompl.hom_eq_iff_le _ _ _).mp e vD, vD⟩


/-- For a locally finite preconnected graph, the number of components outside of any finite set
is finite. -/
instance componentCompl_finite [LocallyFinite G] [Gpc : Fact G.Preconnected] (K : Finset V) :
    Finite (G.ComponentCompl K) := by
  classical
  rcases K.eq_empty_or_nonempty with rfl | h
  -- If K is empty, then removing K doesn't change the graph, which is connected, hence has a
  -- single connected component
  · dsimp [ComponentCompl]
    rw [Finset.coe_empty, Set.compl_empty]
    have := Gpc.out.subsingleton_connectedComponent
    exact Finite.of_equiv _ (induceUnivIso G).connectedComponentEquiv.symm
  -- Otherwise, we consider the function `touch` mapping a connected component to one of its
  -- vertices adjacent to `K`.
  · let touch (C : G.ComponentCompl K) : {v : V | ∃ k : V, k ∈ K ∧ G.Adj k v} :=
      let p := C.exists_adj_boundary_pair Gpc.out h
      ⟨p.choose.1, p.choose.2, p.choose_spec.2.1, p.choose_spec.2.2.symm⟩
    -- `touch` is injective
    have touch_inj : touch.Injective := fun C D h' => ComponentCompl.pairwise_disjoint.eq
      (Set.not_disjoint_iff.mpr ⟨touch C, (C.exists_adj_boundary_pair Gpc.out h).choose_spec.1,
                                 h'.symm ▸ (D.exists_adj_boundary_pair Gpc.out h).choose_spec.1⟩)
    -- `touch` has finite range
    have : Finite (Set.range touch) := by
      refine @Subtype.finite _ (Set.Finite.to_subtype ?_) _
      apply Set.Finite.ofFinset (K.biUnion (fun v => G.neighborFinset v))
      simp only [Finset.mem_biUnion, mem_neighborFinset, Set.mem_setOf_eq, implies_true]
    -- hence `touch` has a finite domain
    apply Finite.of_injective_finite_range touch_inj


/--
The functor assigning, to a finite set in `V`, the set of connected components in its complement.
-/
@[simps]
def componentComplFunctor : (Finset V)ᵒᵖ ⥤ Type u where
  obj K := G.ComponentCompl K.unop
  map f := ComponentCompl.hom (le_of_op_hom f)
  map_id _ := funext fun C => C.hom_refl
  map_comp {_ Y Z} h h' := funext fun C => by
    /-
      V : Type u
      G : SimpleGraph V
      K L M : Set V
      x✝ Y Z : Opposite (Finset V)
      h : Quiver.Hom x✝ Y
      h' : Quiver.Hom Y Z
      C : { obj := fun K => G.ComponentCompl ↑(Opposite.unop K), map := fun {X Y} f  …
      ⊢ Eq ({ obj := fun K => G.ComponentCompl ↑(Opposite.unop K), map := fun {X Y}  …
    -/
    convert C.hom_trans (le_of_op_hom h) (le_of_op_hom _)
    /-
      case convert_2
      V : Type u
      G : SimpleGraph V
      K L M : Set V
      x✝ Y Z : Opposite (Finset V)
      h : Quiver.Hom x✝ Y
      h' : Quiver.Hom Y Z
      C : { obj := fun K => G.ComponentCompl ↑(Opposite.unop K), map := fun {X Y} f  …
      ⊢ Quiver.Hom { unop := Membership.mem (Opposite.unop Y).val } { unop := ↑(Oppo …
    -/
    exact h'
    /-
      🎉 no goals
    -/


/-- The end of a graph, defined as the sections of the functor `component_compl_functor` . -/
protected def «end» :=
  (componentComplFunctor G).sections


theorem end_hom_mk_of_mk {s} (sec : s ∈ G.end) {K L : (Finset V)ᵒᵖ} (h : L ⟶ K) {v : V}
    (vnL : v ∉ L.unop) (hs : s L = G.componentComplMk vnL) :
    s K = G.componentComplMk (Set.not_mem_subset (le_of_op_hom h : _ ⊆ _) vnL) := by
  /-
    V : Type u
    G : SimpleGraph V
    s : (j : Opposite (Finset V)) → G.componentComplFunctor.obj j
    sec : Membership.mem G.end s
    K L : Opposite (Finset V)
    h : Quiver.Hom L K
    v : V
    vnL : Not (Membership.mem (Opposite.unop L) v)
    hs : Eq (s L) (G.componentComplMk vnL)
    ⊢ Eq (s K) (G.componentComplMk ⋯)
  -/
  rw [← sec h, hs]
  /-
    V : Type u
    G : SimpleGraph V
    s : (j : Opposite (Finset V)) → G.componentComplFunctor.obj j
    sec : Membership.mem G.end s
    K L : Opposite (Finset V)
    h : Quiver.Hom L K
    v : V
    vnL : Not (Membership.mem (Opposite.unop L) v)
    hs : Eq (s L) (G.componentComplMk vnL)
    ⊢ Eq (G.componentComplFunctor.map h (G.componentComplMk vnL)) (G.componentComp …
  -/
  apply ComponentCompl.hom_mk _ (le_of_op_hom h : _ ⊆ _)
  /-
    🎉 no goals
  -/


theorem infinite_iff_in_eventualRange {K : (Finset V)ᵒᵖ} (C : G.componentComplFunctor.obj K) :
    C.supp.Infinite ↔ C ∈ G.componentComplFunctor.eventualRange K := by
  simp only [C.infinite_iff_in_all_ranges, CategoryTheory.Functor.eventualRange, Set.mem_iInter,
    Set.mem_range, componentComplFunctor_map]
  exact
    ⟨fun h Lop KL => h Lop.unop (le_of_op_hom KL), fun h L KL =>
      h (Opposite.op L) (opHomOfLE KL)⟩


