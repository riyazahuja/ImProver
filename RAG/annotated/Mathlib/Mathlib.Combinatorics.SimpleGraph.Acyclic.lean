/-- A graph is *acyclic* (or a *forest*) if it has no cycles. -/
def IsAcyclic : Prop := ∀ ⦃v : V⦄ (c : G.Walk v v), ¬c.IsCycle


/-- A *tree* is a connected acyclic graph. -/
@[mk_iff]
structure IsTree : Prop where
  /-- Graph is connected. -/
  protected isConnected : G.Connected
  /-- Graph is acyclic. -/
  protected IsAcyclic : G.IsAcyclic


@[simp] lemma isAcyclic_bot : IsAcyclic (⊥ : SimpleGraph V) := fun _a _w hw ↦ hw.ne_bot rfl


theorem isAcyclic_iff_forall_adj_isBridge :
    G.IsAcyclic ↔ ∀ ⦃v w : V⦄, G.Adj v w → G.IsBridge s(v, w) := by
  /-
    V : Type u
    G : SimpleGraph V
    ⊢ Iff G.IsAcyclic (∀ ⦃v w : V⦄, G.Adj v w → G.IsBridge (Sym2.mk { fst := v, sn …
  -/
  simp_rw [isBridge_iff_adj_and_forall_cycle_not_mem]
  /-
    V : Type u
    G : SimpleGraph V
    ⊢ Iff G.IsAcyclic (∀ ⦃v w : V⦄, G.Adj v w → And (G.Adj v w) (∀ ⦃u : V⦄ (p : G. …
  -/
  constructor
    /-
      case mp
      V : Type u
      G : SimpleGraph V
      ⊢ G.IsAcyclic → ∀ ⦃v w : V⦄, G.Adj v w → And (G.Adj v w) (∀ ⦃u : V⦄ (p : G.Wal …
    -/
  · intro ha v w hvw
    /-
      case mp
      V : Type u
      G : SimpleGraph V
      ha : G.IsAcyclic
      v w : V
      hvw : G.Adj v w
      ⊢ And (G.Adj v w) (∀ ⦃u : V⦄ (p : G.Walk u u), p.IsCycle → Not (Membership.mem …
    -/
    apply And.intro hvw
    /-
      case mp
      V : Type u
      G : SimpleGraph V
      ha : G.IsAcyclic
      v w : V
      hvw : G.Adj v w
      ⊢ ∀ ⦃u : V⦄ (p : G.Walk u u), p.IsCycle → Not (Membership.mem p.edges (Sym2.mk …
    -/
    intro u p hp
    /-
      case mp
      V : Type u
      G : SimpleGraph V
      ha : G.IsAcyclic
      v w : V
      hvw : G.Adj v w
      u : V
      p : G.Walk u u
      hp : p.IsCycle
      ⊢ Not (Membership.mem p.edges (Sym2.mk { fst := v, snd := w }))
    -/
    cases ha p hp
    /-
      🎉 no goals
    -/
    /-
      case mpr
      V : Type u
      G : SimpleGraph V
      ⊢ (∀ ⦃v w : V⦄, G.Adj v w → And (G.Adj v w) (∀ ⦃u : V⦄ (p : G.Walk u u), p.IsC …
    -/
  · rintro hb v (_ | ⟨ha, p⟩) hp
      /-
        case mpr.nil
        V : Type u
        G : SimpleGraph V
        hb : ∀ ⦃v w : V⦄, G.Adj v w → And (G.Adj v w) (∀ ⦃u : V⦄ (p : G.Walk u u), p.I …
        v : V
        hp : SimpleGraph.Walk.nil.IsCycle
        ⊢ False
      -/
    · exact hp.not_of_nil
      /-
        🎉 no goals
      -/
      /-
        case mpr.cons
        V : Type u
        G : SimpleGraph V
        hb : ∀ ⦃v w : V⦄, G.Adj v w → And (G.Adj v w) (∀ ⦃u : V⦄ (p : G.Walk u u), p.I …
        v v✝ : V
        ha : G.Adj v v✝
        p : G.Walk v✝ v
        hp : (SimpleGraph.Walk.cons ha p).IsCycle
        ⊢ False
      -/
    · apply (hb ha).2 _ hp
      /-
        case mpr.cons
        V : Type u
        G : SimpleGraph V
        hb : ∀ ⦃v w : V⦄, G.Adj v w → And (G.Adj v w) (∀ ⦃u : V⦄ (p : G.Walk u u), p.I …
        v v✝ : V
        ha : G.Adj v v✝
        p : G.Walk v✝ v
        hp : (SimpleGraph.Walk.cons ha p).IsCycle
        ⊢ Membership.mem (SimpleGraph.Walk.cons ha p).edges (Sym2.mk { fst := v, snd : …
      -/
      rw [Walk.edges_cons]
      /-
        case mpr.cons
        V : Type u
        G : SimpleGraph V
        hb : ∀ ⦃v w : V⦄, G.Adj v w → And (G.Adj v w) (∀ ⦃u : V⦄ (p : G.Walk u u), p.I …
        v v✝ : V
        ha : G.Adj v v✝
        p : G.Walk v✝ v
        hp : (SimpleGraph.Walk.cons ha p).IsCycle
        ⊢ Membership.mem (List.cons (Sym2.mk { fst := v, snd := v✝ }) p.edges) (Sym2.m …
      -/
      apply List.mem_cons_self
      /-
        🎉 no goals
      -/


theorem isAcyclic_iff_forall_edge_isBridge :
    G.IsAcyclic ↔ ∀ ⦃e⦄, e ∈ (G.edgeSet) → G.IsBridge e := by
  /-
    V : Type u
    G : SimpleGraph V
    ⊢ Iff G.IsAcyclic (∀ ⦃e : Sym2 V⦄, Membership.mem G.edgeSet e → G.IsBridge e)
  -/
  simp [isAcyclic_iff_forall_adj_isBridge, Sym2.forall]
  /-
    🎉 no goals
  -/


theorem IsAcyclic.path_unique {G : SimpleGraph V} (h : G.IsAcyclic) {v w : V} (p q : G.Path v w) :
    p = q := by
  /-
    V : Type u
    G : SimpleGraph V
    h : G.IsAcyclic
    v w : V
    p q : G.Path v w
    ⊢ Eq p q
  -/
  obtain ⟨p, hp⟩ := p
  /-
    case mk
    V : Type u
    G : SimpleGraph V
    h : G.IsAcyclic
    v w : V
    q : G.Path v w
    p : G.Walk v w
    hp : p.IsPath
    ⊢ Eq ⟨p, hp⟩ q
  -/
  obtain ⟨q, hq⟩ := q
  /-
    case mk.mk
    V : Type u
    G : SimpleGraph V
    h : G.IsAcyclic
    v w : V
    p : G.Walk v w
    hp : p.IsPath
    q : G.Walk v w
    hq : q.IsPath
    ⊢ Eq ⟨p, hp⟩ ⟨q, hq⟩
  -/
  rw [Subtype.mk.injEq]
  induction p with
  | nil =>
    cases (Walk.isPath_iff_eq_nil _).mp hq
    rfl
  | cons ph p ih =>
    rw [isAcyclic_iff_forall_adj_isBridge] at h
    specialize h ph
    rw [isBridge_iff_adj_and_forall_walk_mem_edges] at h
    replace h := h.2 (q.append p.reverse)
    simp only [Walk.edges_append, Walk.edges_reverse, List.mem_append, List.mem_reverse] at h
    cases' h with h h
    · cases q with
      | nil => simp [Walk.isPath_def] at hp
      | cons _ q =>
        rw [Walk.cons_isPath_iff] at hp hq
        simp only [Walk.edges_cons, List.mem_cons, Sym2.eq_iff, true_and] at h
        rcases h with (⟨h, rfl⟩ | ⟨rfl, rfl⟩) | h
        · cases ih hp.1 q hq.1
          rfl
        · simp at hq
        · exact absurd (Walk.fst_mem_support_of_mem_edges _ h) hq.2
    · rw [Walk.cons_isPath_iff] at hp
      exact absurd (Walk.fst_mem_support_of_mem_edges _ h) hp.2


theorem isAcyclic_of_path_unique (h : ∀ (v w : V) (p q : G.Path v w), p = q) : G.IsAcyclic := by
  /-
    V : Type u
    G : SimpleGraph V
    h : ∀ (v w : V) (p q : G.Path v w), Eq p q
    ⊢ G.IsAcyclic
  -/
  intro v c hc
  /-
    V : Type u
    G : SimpleGraph V
    h : ∀ (v w : V) (p q : G.Path v w), Eq p q
    v : V
    c : G.Walk v v
    hc : c.IsCycle
    ⊢ False
  -/
  simp only [Walk.isCycle_def, Ne] at hc
  cases c with
  | nil => cases hc.2.1 rfl
  | cons ha c' =>
    simp only [Walk.cons_isTrail_iff, Walk.support_cons, List.tail_cons] at hc
    specialize h _ _ ⟨c', by simp only [Walk.isPath_def, hc.2]⟩ (Path.singleton ha.symm)
    rw [Path.singleton, Subtype.mk.injEq] at h
    simp [h] at hc


theorem isAcyclic_iff_path_unique : G.IsAcyclic ↔ ∀ ⦃v w : V⦄ (p q : G.Path v w), p = q :=
  ⟨IsAcyclic.path_unique, isAcyclic_of_path_unique⟩


theorem isTree_iff_existsUnique_path :
    G.IsTree ↔ Nonempty V ∧ ∀ v w : V, ∃! p : G.Walk v w, p.IsPath := by
  classical
  rw [isTree_iff, isAcyclic_iff_path_unique]
  constructor
  · rintro ⟨hc, hu⟩
    refine ⟨hc.nonempty, ?_⟩
    intro v w
    let q := (hc v w).some.toPath
    use q
    simp only [true_and, Path.isPath]
    intro p hp
    specialize hu ⟨p, hp⟩ q
    exact Subtype.ext_iff.mp hu
  · rintro ⟨hV, h⟩
    refine ⟨Connected.mk ?_, ?_⟩
    · intro v w
      obtain ⟨p, _⟩ := h v w
      exact p.reachable
    · rintro v w ⟨p, hp⟩ ⟨q, hq⟩
      simp only [ExistsUnique.unique (h v w) hp hq]


lemma IsTree.existsUnique_path (hG : G.IsTree) : ∀ v w, ∃! p : G.Walk v w, p.IsPath :=
  (isTree_iff_existsUnique_path.1 hG).2


lemma IsTree.card_edgeFinset [Fintype V] [Fintype G.edgeSet] (hG : G.IsTree) :
    Finset.card G.edgeFinset + 1 = Fintype.card V := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝¹ : Fintype V
    inst✝ : Fintype ↑G.edgeSet
    hG : G.IsTree
    ⊢ Eq (HAdd.hAdd G.edgeFinset.card 1) (Fintype.card V)
  -/
  have := hG.isConnected.nonempty
  /-
    V : Type u
    G : SimpleGraph V
    inst✝¹ : Fintype V
    inst✝ : Fintype ↑G.edgeSet
    hG : G.IsTree
    this : Nonempty V
    ⊢ Eq (HAdd.hAdd G.edgeFinset.card 1) (Fintype.card V)
  -/
  inhabit V
  classical
  have : Finset.card ({default} : Finset V)ᶜ + 1 = Fintype.card V := by
    rw [Finset.card_compl, Finset.card_singleton, Nat.sub_add_cancel Fintype.card_pos]
  rw [← this, add_left_inj]
  choose f hf hf' using (hG.existsUnique_path · default)
  refine Eq.symm <| Finset.card_bij
          (fun w hw => ((f w).firstDart <| ?notNil).edge)
          (fun a ha => ?memEdges) ?inj ?surj
  case notNil => exact not_nil_of_ne (by simpa using hw)
  case memEdges => simp
  case inj =>
    intros a ha b hb h
    wlog h' : (f a).length ≤ (f b).length generalizing a b
    · exact Eq.symm (this _ hb _ ha h.symm (le_of_not_le h'))
    rw [dart_edge_eq_iff] at h
    obtain (h | h) := h
    · exact (congrArg (·.fst) h)
    · have h1 : ((f a).firstDart <| not_nil_of_ne (by simpa using ha)).snd = b :=
        congrArg (·.snd) h
      have h3 := congrArg length (hf' _ ((f _).tail.copy h1 rfl) ?_)
      · rw [length_copy, ← add_left_inj 1,
          length_tail_add_one (not_nil_of_ne (by simpa using ha))] at h3
        omega
      · simp only [ne_eq, eq_mp_eq_cast, id_eq, isPath_copy]
        exact (hf _).tail (not_nil_of_ne (by simpa using ha))
  case surj =>
    simp only [mem_edgeFinset, Finset.mem_compl, Finset.mem_singleton, Sym2.forall, mem_edgeSet]
    intros x y h
    wlog h' : (f x).length ≤ (f y).length generalizing x y
    · rw [Sym2.eq_swap]
      exact this y x h.symm (le_of_not_le h')
    refine ⟨y, ?_, dart_edge_eq_mk'_iff.2 <| Or.inr ?_⟩
    · rintro rfl
      rw [← hf' _ nil IsPath.nil, length_nil,
          ← hf' _ (.cons h .nil) (IsPath.nil.cons <| by simpa using h.ne),
          length_cons, length_nil] at h'
      simp [Nat.le_zero, Nat.one_ne_zero] at h'
    rw [← hf' _ (.cons h.symm (f x)) ((cons_isPath_iff _ _).2 ⟨hf _, fun hy => ?contra⟩)]
    · simp only [firstDart_toProd, getVert_cons_succ, getVert_zero, Prod.swap_prod_mk]
    case contra =>
      suffices (f x).takeUntil y hy = .cons h .nil by
        rw [← take_spec _ hy] at h'
        simp [this, hf' _ _ ((hf _).dropUntil hy)] at h'
      refine (hG.existsUnique_path _ _).unique ((hf _).takeUntil _) ?_
      simp [h.ne]


