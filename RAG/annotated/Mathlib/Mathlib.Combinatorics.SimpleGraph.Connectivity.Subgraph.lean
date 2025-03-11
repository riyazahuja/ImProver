/-- A subgraph is preconnected if it is preconnected when coerced to be a simple graph.

Note: This is a structure to make it so one can be precise about how dot notation resolves. -/
protected structure Preconnected (H : G.Subgraph) : Prop where
  protected coe : H.coe.Preconnected


instance {H : G.Subgraph} : Coe H.Preconnected H.coe.Preconnected := ⟨Preconnected.coe⟩


instance {H : G.Subgraph} : CoeFun H.Preconnected (fun _ => ∀ u v : H.verts, H.coe.Reachable u v) :=
  ⟨fun h => h.coe⟩


protected lemma preconnected_iff {H : G.Subgraph} :
    H.Preconnected ↔ H.coe.Preconnected := ⟨fun ⟨h⟩ => h, .mk⟩


/-- A subgraph is connected if it is connected when coerced to be a simple graph.

Note: This is a structure to make it so one can be precise about how dot notation resolves. -/
protected structure Connected (H : G.Subgraph) : Prop where
  protected coe : H.coe.Connected


instance {H : G.Subgraph} : Coe H.Connected H.coe.Connected := ⟨Connected.coe⟩


instance {H : G.Subgraph} : CoeFun H.Connected (fun _ => ∀ u v : H.verts, H.coe.Reachable u v) :=
  ⟨fun h => h.coe⟩


protected lemma connected_iff' {H : G.Subgraph} :
    H.Connected ↔ H.coe.Connected := ⟨fun ⟨h⟩ => h, .mk⟩


protected lemma connected_iff {H : G.Subgraph} :
    H.Connected ↔ H.Preconnected ∧ H.verts.Nonempty := by
  /-
    V : Type u
    G : SimpleGraph V
    H : G.Subgraph
    ⊢ Iff H.Connected (And H.Preconnected H.verts.Nonempty)
  -/
  rw [H.connected_iff', connected_iff, H.preconnected_iff, Set.nonempty_coe_sort]
  /-
    🎉 no goals
  -/


protected lemma Connected.preconnected {H : G.Subgraph} (h : H.Connected) : H.Preconnected := by
  /-
    V : Type u
    G : SimpleGraph V
    H : G.Subgraph
    h : H.Connected
    ⊢ H.Preconnected
  -/
  rw [H.connected_iff] at h; exact h.1
                             /-
                               🎉 no goals
                             -/


protected lemma Connected.nonempty {H : G.Subgraph} (h : H.Connected) : H.verts.Nonempty := by
  /-
    V : Type u
    G : SimpleGraph V
    H : G.Subgraph
    h : H.Connected
    ⊢ H.verts.Nonempty
  -/
  rw [H.connected_iff] at h; exact h.2
                             /-
                               🎉 no goals
                             -/


theorem singletonSubgraph_connected {v : V} : (G.singletonSubgraph v).Connected := by
  /-
    V : Type u
    G : SimpleGraph V
    v : V
    ⊢ (G.singletonSubgraph v).Connected
  -/
  refine ⟨⟨?_⟩⟩
  /-
    V : Type u
    G : SimpleGraph V
    v : V
    ⊢ (G.singletonSubgraph v).coe.Preconnected
  -/
  rintro ⟨a, ha⟩ ⟨b, hb⟩
  /-
    case mk.mk
    V : Type u
    G : SimpleGraph V
    v a : V
    ha : Membership.mem (G.singletonSubgraph v).verts a
    b : V
    hb : Membership.mem (G.singletonSubgraph v).verts b
    ⊢ (G.singletonSubgraph v).coe.Reachable ⟨a, ha⟩ ⟨b, hb⟩
  -/
  simp only [singletonSubgraph_verts, Set.mem_singleton_iff] at ha hb
  /-
    case mk.mk
    V : Type u
    G : SimpleGraph V
    v a : V
    ha✝ : Membership.mem (G.singletonSubgraph v).verts a
    b : V
    hb✝ : Membership.mem (G.singletonSubgraph v).verts b
    ha : Eq a v
    hb : Eq b v
    ⊢ (G.singletonSubgraph v).coe.Reachable ⟨a, ha✝⟩ ⟨b, hb✝⟩
  -/
  subst_vars
  /-
    case mk.mk
    V : Type u
    G : SimpleGraph V
    b : V
    ha hb : Eq b b
    ⊢ (G.singletonSubgraph b).coe.Reachable ⟨b, ⋯⟩ ⟨b, ⋯⟩
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem subgraphOfAdj_connected {v w : V} (hvw : G.Adj v w) : (G.subgraphOfAdj hvw).Connected := by
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    hvw : G.Adj v w
    ⊢ (G.subgraphOfAdj hvw).Connected
  -/
  refine ⟨⟨?_⟩⟩
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    hvw : G.Adj v w
    ⊢ (G.subgraphOfAdj hvw).coe.Preconnected
  -/
  rintro ⟨a, ha⟩ ⟨b, hb⟩
  /-
    case mk.mk
    V : Type u
    G : SimpleGraph V
    v w : V
    hvw : G.Adj v w
    a : V
    ha : Membership.mem (G.subgraphOfAdj hvw).verts a
    b : V
    hb : Membership.mem (G.subgraphOfAdj hvw).verts b
    ⊢ (G.subgraphOfAdj hvw).coe.Reachable ⟨a, ha⟩ ⟨b, hb⟩
  -/
  simp only [subgraphOfAdj_verts, Set.mem_insert_iff, Set.mem_singleton_iff] at ha hb
  /-
    case mk.mk
    V : Type u
    G : SimpleGraph V
    v w : V
    hvw : G.Adj v w
    a : V
    ha✝ : Membership.mem (G.subgraphOfAdj hvw).verts a
    b : V
    hb✝ : Membership.mem (G.subgraphOfAdj hvw).verts b
    ha : Or (Eq a v) (Eq a w)
    hb : Or (Eq b v) (Eq b w)
    ⊢ (G.subgraphOfAdj hvw).coe.Reachable ⟨a, ha✝⟩ ⟨b, hb✝⟩
  -/
  obtain rfl | rfl := ha <;> obtain rfl | rfl := hb <;>
    /-
      case mk.mk.inl.inl
      V : Type u
      G : SimpleGraph V
      w b : V
      hvw : G.Adj b w
      ha hb : Membership.mem (G.subgraphOfAdj hvw).verts b
      ⊢ (G.subgraphOfAdj hvw).coe.Reachable ⟨b, ha⟩ ⟨b, hb⟩
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    first | rfl | (apply Adj.reachable; simp)
    /-
      🎉 no goals
    -/


lemma top_induce_pair_connected_of_adj {u v : V} (huv : G.Adj u v) :
    ((⊤ : G.Subgraph).induce {u, v}).Connected := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    huv : G.Adj u v
    ⊢ (Top.top.induce (Insert.insert u (Singleton.singleton v))).Connected
  -/
  rw [← subgraphOfAdj_eq_induce huv]
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    huv : G.Adj u v
    ⊢ (G.subgraphOfAdj huv).Connected
  -/
  exact subgraphOfAdj_connected huv
  /-
    🎉 no goals
  -/


@[mono]
protected lemma Connected.mono {H H' : G.Subgraph} (hle : H ≤ H') (hv : H.verts = H'.verts)
    (h : H.Connected) : H'.Connected := by
  /-
    V : Type u
    G : SimpleGraph V
    H H' : G.Subgraph
    hle : LE.le H H'
    hv : Eq H.verts H'.verts
    h : H.Connected
    ⊢ H'.Connected
  -/
  rw [← Subgraph.copy_eq H' H.verts hv H'.Adj rfl]
  /-
    V : Type u
    G : SimpleGraph V
    H H' : G.Subgraph
    hle : LE.le H H'
    hv : Eq H.verts H'.verts
    h : H.Connected
    ⊢ (H'.copy H.verts hv H'.Adj ⋯).Connected
  -/
  refine ⟨h.coe.mono ?_⟩
  /-
    V : Type u
    G : SimpleGraph V
    H H' : G.Subgraph
    hle : LE.le H H'
    hv : Eq H.verts H'.verts
    h : H.Connected
    ⊢ LE.le H.coe (H'.copy H.verts hv H'.Adj ⋯).coe
  -/
  rintro ⟨v, hv⟩ ⟨w, hw⟩ hvw
  /-
    case mk.mk
    V : Type u
    G : SimpleGraph V
    H H' : G.Subgraph
    hle : LE.le H H'
    hv✝ : Eq H.verts H'.verts
    h : H.Connected
    v : V
    hv : Membership.mem H.verts v
    w : V
    hw : Membership.mem H.verts w
    hvw : H.coe.Adj ⟨v, hv⟩ ⟨w, hw⟩
    ⊢ (H'.copy H.verts hv✝ H'.Adj ⋯).coe.Adj ⟨v, hv⟩ ⟨w, hw⟩
  -/
  exact hle.2 hvw
  /-
    🎉 no goals
  -/


protected lemma Connected.mono' {H H' : G.Subgraph}
    (hle : ∀ v w, H.Adj v w → H'.Adj v w) (hv : H.verts = H'.verts)
    (h : H.Connected) : H'.Connected := by
  /-
    V : Type u
    G : SimpleGraph V
    H H' : G.Subgraph
    hle : ∀ (v w : V), H.Adj v w → H'.Adj v w
    hv : Eq H.verts H'.verts
    h : H.Connected
    ⊢ H'.Connected
  -/
  exact h.mono ⟨hv.le, hle⟩ hv
  /-
    🎉 no goals
  -/


protected lemma Connected.sup {H K : G.Subgraph}
    (hH : H.Connected) (hK : K.Connected) (hn : (H ⊓ K).verts.Nonempty) :
    (H ⊔ K).Connected := by
  /-
    V : Type u
    G : SimpleGraph V
    H K : G.Subgraph
    hH : H.Connected
    hK : K.Connected
    hn : (Min.min H K).verts.Nonempty
    ⊢ (Max.max H K).Connected
  -/
  rw [Subgraph.connected_iff', connected_iff_exists_forall_reachable]
  /-
    V : Type u
    G : SimpleGraph V
    H K : G.Subgraph
    hH : H.Connected
    hK : K.Connected
    hn : (Min.min H K).verts.Nonempty
    ⊢ Exists fun v => ∀ (w : ↑(Max.max H K).verts), (Max.max H K).coe.Reachable v w
  -/
  obtain ⟨u, hu, hu'⟩ := hn
  /-
    case intro.intro
    V : Type u
    G : SimpleGraph V
    H K : G.Subgraph
    hH : H.Connected
    hK : K.Connected
    u : V
    hu : Membership.mem H.verts u
    hu' : Membership.mem K.verts u
    ⊢ Exists fun v => ∀ (w : ↑(Max.max H K).verts), (Max.max H K).coe.Reachable v w
  -/
  exists ⟨u, Or.inl hu⟩
  /-
    case intro.intro
    V : Type u
    G : SimpleGraph V
    H K : G.Subgraph
    hH : H.Connected
    hK : K.Connected
    u : V
    hu : Membership.mem H.verts u
    hu' : Membership.mem K.verts u
    ⊢ ∀ (w : ↑(Max.max H K).verts), (Max.max H K).coe.Reachable ⟨u, ⋯⟩ w
  -/
  rintro ⟨v, (hv|hv)⟩
    /-
      case intro.intro.mk.inl
      V : Type u
      G : SimpleGraph V
      H K : G.Subgraph
      hH : H.Connected
      hK : K.Connected
      u : V
      hu : Membership.mem H.verts u
      hu' : Membership.mem K.verts u
      v : V
      hv : Membership.mem H.verts v
      ⊢ (Max.max H K).coe.Reachable ⟨u, ⋯⟩ ⟨v, ⋯⟩
    -/
  · exact Reachable.map (Subgraph.inclusion (le_sup_left : H ≤ H ⊔ K)) (hH ⟨u, hu⟩ ⟨v, hv⟩)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.mk.inr
      V : Type u
      G : SimpleGraph V
      H K : G.Subgraph
      hH : H.Connected
      hK : K.Connected
      u : V
      hu : Membership.mem H.verts u
      hu' : Membership.mem K.verts u
      v : V
      hv : Membership.mem K.verts v
      ⊢ (Max.max H K).coe.Reachable ⟨u, ⋯⟩ ⟨v, ⋯⟩
    -/
  · exact Reachable.map (Subgraph.inclusion (le_sup_right : K ≤ H ⊔ K)) (hK ⟨u, hu'⟩ ⟨v, hv⟩)
    /-
      🎉 no goals
    -/

/-- The subgraph consisting of the vertices and edges of the walk. -/
@[simp]
protected def toSubgraph {u v : V} : G.Walk u v → G.Subgraph
  | nil => G.singletonSubgraph u
  | cons h p => G.subgraphOfAdj h ⊔ p.toSubgraph


theorem toSubgraph_cons_nil_eq_subgraphOfAdj (h : G.Adj u v) :
                                                      /-
                                                        V : Type u
                                                        G : SimpleGraph V
                                                        u v : V
                                                        h : G.Adj u v
                                                        ⊢ Eq (SimpleGraph.Walk.cons h SimpleGraph.Walk.nil).toSubgraph (G.subgraphOfAd …
                                                      -/
    (cons h nil).toSubgraph = G.subgraphOfAdj h := by simp
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem mem_verts_toSubgraph (p : G.Walk u v) : w ∈ p.toSubgraph.verts ↔ w ∈ p.support := by
  /-
    V : Type u
    G : SimpleGraph V
    u v w : V
    p : G.Walk u v
    ⊢ Iff (Membership.mem p.toSubgraph.verts w) (Membership.mem p.support w)
  -/
  induction' p with _ x y z h p' ih
    /-
      case nil
      V : Type u
      G : SimpleGraph V
      u v w u✝ : V
      ⊢ Iff (Membership.mem SimpleGraph.Walk.nil.toSubgraph.verts w) (Membership.mem …
    -/
  · simp
    /-
      🎉 no goals
    -/
  · have : w = y ∨ w ∈ p'.support ↔ w ∈ p'.support :=
      ⟨by rintro (rfl | h) <;> simp [*], by simp +contextual⟩
    /-
      case cons
      V : Type u
      G : SimpleGraph V
      u v w x y z : V
      h : G.Adj x y
      p' : G.Walk y z
      ih : Iff (Membership.mem p'.toSubgraph.verts w) (Membership.mem p'.support w)
      this : Iff (Or (Eq w y) (Membership.mem p'.support w)) (Membership.mem p'.supp …
      ⊢ Iff (Membership.mem (SimpleGraph.Walk.cons h p').toSubgraph.verts w) (Member …
    -/
    simp [ih, or_assoc, this]
    /-
      🎉 no goals
    -/


lemma start_mem_verts_toSubgraph (p : G.Walk u v) : u ∈ p.toSubgraph.verts := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    ⊢ Membership.mem p.toSubgraph.verts u
  -/
  simp [mem_verts_toSubgraph]
  /-
    🎉 no goals
  -/


lemma end_mem_verts_toSubgraph (p : G.Walk u v) : v ∈ p.toSubgraph.verts := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    ⊢ Membership.mem p.toSubgraph.verts v
  -/
  simp [mem_verts_toSubgraph]
  /-
    🎉 no goals
  -/


@[simp]
theorem verts_toSubgraph (p : G.Walk u v) : p.toSubgraph.verts = { w | w ∈ p.support } :=
  Set.ext fun _ => p.mem_verts_toSubgraph


theorem mem_edges_toSubgraph (p : G.Walk u v) {e : Sym2 V} :
                                                 /-
                                                   V : Type u
                                                   G : SimpleGraph V
                                                   u v : V
                                                   p : G.Walk u v
                                                   e : Sym2 V
                                                   ⊢ Iff (Membership.mem p.toSubgraph.edgeSet e) (Membership.mem p.edges e)
                                                 -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
    e ∈ p.toSubgraph.edgeSet ↔ e ∈ p.edges := by induction p <;> simp [*]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp]
theorem edgeSet_toSubgraph (p : G.Walk u v) : p.toSubgraph.edgeSet = { e | e ∈ p.edges } :=
  Set.ext fun _ => p.mem_edges_toSubgraph


@[simp]
theorem toSubgraph_append (p : G.Walk u v) (q : G.Walk v w) :
                                                                /-
                                                                  V : Type u
                                                                  G : SimpleGraph V
                                                                  u v w : V
                                                                  p : G.Walk u v
                                                                  q : G.Walk v w
                                                                  ⊢ Eq (p.append q).toSubgraph (Max.max p.toSubgraph q.toSubgraph)
                                                                -/
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
    (p.append q).toSubgraph = p.toSubgraph ⊔ q.toSubgraph := by induction p <;> simp [*, sup_assoc]
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


@[simp]
theorem toSubgraph_reverse (p : G.Walk u v) : p.reverse.toSubgraph = p.toSubgraph := by
  induction p with
  | nil => simp
  | cons _ _ _ =>
    simp only [*, Walk.toSubgraph, reverse_cons, toSubgraph_append, subgraphOfAdj_symm]
    rw [sup_comm]
    congr
    ext <;> simp [-Set.bot_eq_empty]


@[simp]
theorem toSubgraph_rotate [DecidableEq V] (c : G.Walk v v) (h : u ∈ c.support) :
    (c.rotate h).toSubgraph = c.toSubgraph := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    inst✝ : DecidableEq V
    c : G.Walk v v
    h : Membership.mem c.support u
    ⊢ Eq (c.rotate h).toSubgraph c.toSubgraph
  -/
  rw [rotate, toSubgraph_append, sup_comm, ← toSubgraph_append, take_spec]
  /-
    🎉 no goals
  -/


@[simp]
theorem toSubgraph_map (f : G →g G') (p : G.Walk u v) :
                                                    /-
                                                      V : Type u
                                                      V' : Type v
                                                      G : SimpleGraph V
                                                      G' : SimpleGraph V'
                                                      u v : V
                                                      f : G.Hom G'
                                                      p : G.Walk u v
                                                      ⊢ Eq (SimpleGraph.Walk.map f p).toSubgraph (SimpleGraph.Subgraph.map f p.toSub …
                                                    -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
    (p.map f).toSubgraph = p.toSubgraph.map f := by induction p <;> simp [*, Subgraph.map_sup]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[simp]
theorem finite_neighborSet_toSubgraph (p : G.Walk u v) : (p.toSubgraph.neighborSet w).Finite := by
  induction p with
  | nil =>
    rw [Walk.toSubgraph, neighborSet_singletonSubgraph]
    apply Set.toFinite
  | cons ha _ ih =>
    rw [Walk.toSubgraph, Subgraph.neighborSet_sup]
    refine Set.Finite.union ?_ ih
    refine Set.Finite.subset ?_ (neighborSet_subgraphOfAdj_subset ha)
    apply Set.toFinite


lemma toSubgraph_le_induce_support (p : G.Walk u v) :
    p.toSubgraph ≤ (⊤ : G.Subgraph).induce {v | v ∈ p.support} := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    ⊢ LE.le p.toSubgraph (Top.top.induce (setOf fun v_1 => Membership.mem p.suppor …
  -/
  convert Subgraph.le_induce_top_verts
  /-
    case h.e'_4.h.e'_4
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    ⊢ Eq (setOf fun v_1 => Membership.mem p.support v_1) p.toSubgraph.verts
  -/
  exact p.verts_toSubgraph.symm
  /-
    🎉 no goals
  -/


theorem toSubgraph_adj_getVert {u v} (w : G.Walk u v) {i : ℕ} (hi : i < w.length) :
    w.toSubgraph.Adj (w.getVert i) (w.getVert (i + 1)) := by
  induction w generalizing i with
  | nil => cases hi
  | cons hxy i' ih =>
    cases i
    · simp only [Walk.toSubgraph, Walk.getVert_zero, zero_add, getVert_cons_succ, Subgraph.sup_adj,
      subgraphOfAdj_adj, true_or]
    · simp only [Walk.toSubgraph, getVert_cons_succ, Subgraph.sup_adj, subgraphOfAdj_adj, Sym2.eq,
      Sym2.rel_iff', Prod.mk.injEq, Prod.swap_prod_mk]
      right
      exact ih (Nat.succ_lt_succ_iff.mp hi)


theorem toSubgraph_adj_iff {u v u' v'} (w : G.Walk u v) :
    w.toSubgraph.Adj u' v' ↔ ∃ i, s(w.getVert i, w.getVert (i + 1)) =
      s(u', v') ∧ i < w.length := by
  /-
    V : Type u
    G : SimpleGraph V
    u v u' v' : V
    w : G.Walk u v
    ⊢ Iff (w.toSubgraph.Adj u' v') (Exists fun i => And (Eq (Sym2.mk { fst := w.ge …
  -/
  constructor
    /-
      case mp
      V : Type u
      G : SimpleGraph V
      u v u' v' : V
      w : G.Walk u v
      ⊢ w.toSubgraph.Adj u' v' → Exists fun i => And (Eq (Sym2.mk { fst := w.getVert …
    -/
  · intro hadj
    /-
      case mp
      V : Type u
      G : SimpleGraph V
      u v u' v' : V
      w : G.Walk u v
      hadj : w.toSubgraph.Adj u' v'
      ⊢ Exists fun i => And (Eq (Sym2.mk { fst := w.getVert i, snd := w.getVert (HAd …
    -/
    unfold Walk.toSubgraph at hadj
    match w with
    | .nil =>
      simp only [singletonSubgraph_adj, Pi.bot_apply, Prop.bot_eq_false] at hadj
    | .cons h p =>
      simp only [Subgraph.sup_adj, subgraphOfAdj_adj, Sym2.eq, Sym2.rel_iff', Prod.mk.injEq,
        Prod.swap_prod_mk] at hadj
      cases hadj with
      | inl hl =>
        use 0
        simp only [Walk.getVert_zero, zero_add, getVert_cons_succ]
        refine ⟨?_, by simp only [length_cons, Nat.zero_lt_succ]⟩
        simp only [Sym2.eq, Sym2.rel_iff', Prod.mk.injEq, Prod.swap_prod_mk]
        cases hl with
        | inl h1 => left; exact ⟨h1.1, h1.2⟩
        | inr h2 => right; exact ⟨h2.1, h2.2⟩
      | inr hr =>
        obtain ⟨i, hi⟩ := (toSubgraph_adj_iff _).mp hr
        use i + 1
        simp only [getVert_cons_succ]
        constructor
        · exact hi.1
        · simp only [Walk.length_cons, add_lt_add_iff_right, Nat.add_lt_add_right hi.2 1]
    /-
      case mpr
      V : Type u
      G : SimpleGraph V
      u v u' v' : V
      w : G.Walk u v
      ⊢ (Exists fun i => And (Eq (Sym2.mk { fst := w.getVert i, snd := w.getVert (HA …
    -/
  · rintro ⟨i, hi⟩
    /-
      case mpr.intro
      V : Type u
      G : SimpleGraph V
      u v u' v' : V
      w : G.Walk u v
      i : Nat
      hi : And (Eq (Sym2.mk { fst := w.getVert i, snd := w.getVert (HAdd.hAdd i 1) } …
      ⊢ w.toSubgraph.Adj u' v'
    -/
    rw [← Subgraph.mem_edgeSet, ← hi.1, Subgraph.mem_edgeSet]
    /-
      case mpr.intro
      V : Type u
      G : SimpleGraph V
      u v u' v' : V
      w : G.Walk u v
      i : Nat
      hi : And (Eq (Sym2.mk { fst := w.getVert i, snd := w.getVert (HAdd.hAdd i 1) } …
      ⊢ w.toSubgraph.Adj (w.getVert i) (w.getVert (HAdd.hAdd i 1))
    -/
    exact toSubgraph_adj_getVert _ hi.2
    /-
      🎉 no goals
    -/


lemma _root_.SimpleGraph.Walk.toSubgraph_connected {u v : V} (p : G.Walk u v) :
    p.toSubgraph.Connected := by
  induction p with
  | nil => apply singletonSubgraph_connected
  | @cons _ w _ h p ih =>
    apply (subgraphOfAdj_connected h).sup ih
    exists w
    simp


lemma induce_union_connected {H : G.Subgraph} {s t : Set V}
    (sconn : (H.induce s).Connected) (tconn : (H.induce t).Connected)
    (sintert : (s ⊓ t).Nonempty) :
    (H.induce (s ∪ t)).Connected := by
  /-
    V : Type u
    G : SimpleGraph V
    H : G.Subgraph
    s t : Set V
    sconn : (H.induce s).Connected
    tconn : (H.induce t).Connected
    sintert : (Min.min s t).Nonempty
    ⊢ (H.induce (Union.union s t)).Connected
  -/
  refine (sconn.sup tconn sintert).mono ?_ ?_
    /-
      case refine_1
      V : Type u
      G : SimpleGraph V
      H : G.Subgraph
      s t : Set V
      sconn : (H.induce s).Connected
      tconn : (H.induce t).Connected
      sintert : (Min.min s t).Nonempty
      ⊢ LE.le (Max.max (H.induce s) (H.induce t)) (H.induce (Union.union s t))
    -/
  · apply le_induce_union
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      V : Type u
      G : SimpleGraph V
      H : G.Subgraph
      s t : Set V
      sconn : (H.induce s).Connected
      tconn : (H.induce t).Connected
      sintert : (Min.min s t).Nonempty
      ⊢ Eq (Max.max (H.induce s) (H.induce t)).verts (H.induce (Union.union s t)).ve …
    -/
  · simp
    /-
      🎉 no goals
    -/


lemma Connected.adj_union {H K : G.Subgraph}
    (Hconn : H.Connected) (Kconn : K.Connected) {u v : V} (uH : u ∈ H.verts) (vK : v ∈ K.verts)
    (huv : G.Adj u v) :
    ((⊤ : G.Subgraph).induce {u, v} ⊔ H ⊔ K).Connected := by
  /-
    V : Type u
    G : SimpleGraph V
    H K : G.Subgraph
    Hconn : H.Connected
    Kconn : K.Connected
    u v : V
    uH : Membership.mem H.verts u
    vK : Membership.mem K.verts v
    huv : G.Adj u v
    ⊢ (Max.max (Max.max (Top.top.induce (Insert.insert u (Singleton.singleton v))) …
  -/
  refine ((top_induce_pair_connected_of_adj huv).sup Hconn ?_).sup Kconn ?_
    /-
      case refine_1
      V : Type u
      G : SimpleGraph V
      H K : G.Subgraph
      Hconn : H.Connected
      Kconn : K.Connected
      u v : V
      uH : Membership.mem H.verts u
      vK : Membership.mem K.verts v
      huv : G.Adj u v
      ⊢ (Min.min (Top.top.induce (Insert.insert u (Singleton.singleton v))) H).verts …
    -/
  · exact ⟨u, by simp [uH]⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      V : Type u
      G : SimpleGraph V
      H K : G.Subgraph
      Hconn : H.Connected
      Kconn : K.Connected
      u v : V
      uH : Membership.mem H.verts u
      vK : Membership.mem K.verts v
      huv : G.Adj u v
      ⊢ (Min.min (Max.max (Top.top.induce (Insert.insert u (Singleton.singleton v))) …
    -/
  · exact ⟨v, by simp [vK]⟩
    /-
      🎉 no goals
    -/


lemma preconnected_iff_forall_exists_walk_subgraph (H : G.Subgraph) :
    H.Preconnected ↔ ∀ {u v}, u ∈ H.verts → v ∈ H.verts → ∃ p : G.Walk u v, p.toSubgraph ≤ H := by
  /-
    V : Type u
    G : SimpleGraph V
    H : G.Subgraph
    ⊢ Iff H.Preconnected (∀ {u v : V}, Membership.mem H.verts u → Membership.mem H …
  -/
  constructor
    /-
      case mp
      V : Type u
      G : SimpleGraph V
      H : G.Subgraph
      ⊢ H.Preconnected → ∀ {u v : V}, Membership.mem H.verts u → Membership.mem H.ve …
    -/
  · intro hc u v hu hv
    /-
      case mp
      V : Type u
      G : SimpleGraph V
      H : G.Subgraph
      hc : H.Preconnected
      u v : V
      hu : Membership.mem H.verts u
      hv : Membership.mem H.verts v
      ⊢ Exists fun p => LE.le p.toSubgraph H
    -/
    refine (hc ⟨_, hu⟩ ⟨_, hv⟩).elim fun p => ?_
    /-
      case mp
      V : Type u
      G : SimpleGraph V
      H : G.Subgraph
      hc : H.Preconnected
      u v : V
      hu : Membership.mem H.verts u
      hv : Membership.mem H.verts v
      p : H.coe.Walk ⟨u, hu⟩ ⟨v, hv⟩
      ⊢ Exists fun p => LE.le p.toSubgraph H
    -/
    exists p.map (Subgraph.hom _)
    /-
      case mp
      V : Type u
      G : SimpleGraph V
      H : G.Subgraph
      hc : H.Preconnected
      u v : V
      hu : Membership.mem H.verts u
      hv : Membership.mem H.verts v
      p : H.coe.Walk ⟨u, hu⟩ ⟨v, hv⟩
      ⊢ LE.le (SimpleGraph.Walk.map H.hom p).toSubgraph H
    -/
    simp [coeSubgraph_le]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      V : Type u
      G : SimpleGraph V
      H : G.Subgraph
      ⊢ (∀ {u v : V}, Membership.mem H.verts u → Membership.mem H.verts v → Exists f …
    -/
  · intro hw
    /-
      case mpr
      V : Type u
      G : SimpleGraph V
      H : G.Subgraph
      hw : ∀ {u v : V}, Membership.mem H.verts u → Membership.mem H.verts v → Exists …
      ⊢ H.Preconnected
    -/
    rw [Subgraph.preconnected_iff]
    /-
      case mpr
      V : Type u
      G : SimpleGraph V
      H : G.Subgraph
      hw : ∀ {u v : V}, Membership.mem H.verts u → Membership.mem H.verts v → Exists …
      ⊢ H.coe.Preconnected
    -/
    rintro ⟨u, hu⟩ ⟨v, hv⟩
    /-
      case mpr.mk.mk
      V : Type u
      G : SimpleGraph V
      H : G.Subgraph
      hw : ∀ {u v : V}, Membership.mem H.verts u → Membership.mem H.verts v → Exists …
      u : V
      hu : Membership.mem H.verts u
      v : V
      hv : Membership.mem H.verts v
      ⊢ H.coe.Reachable ⟨u, hu⟩ ⟨v, hv⟩
    -/
    obtain ⟨p, h⟩ := hw hu hv
    exact Reachable.map (Subgraph.inclusion h)
      (p.toSubgraph_connected ⟨_, p.start_mem_verts_toSubgraph⟩ ⟨_, p.end_mem_verts_toSubgraph⟩)


lemma connected_iff_forall_exists_walk_subgraph (H : G.Subgraph) :
    H.Connected ↔
      H.verts.Nonempty ∧
        ∀ {u v}, u ∈ H.verts → v ∈ H.verts → ∃ p : G.Walk u v, p.toSubgraph ≤ H := by
  /-
    V : Type u
    G : SimpleGraph V
    H : G.Subgraph
    ⊢ Iff H.Connected (And H.verts.Nonempty (∀ {u v : V}, Membership.mem H.verts u …
  -/
  rw [H.connected_iff, preconnected_iff_forall_exists_walk_subgraph, and_comm]
  /-
    🎉 no goals
  -/


lemma connected_induce_iff {s : Set V} :
    (G.induce s).Connected ↔ ((⊤ : G.Subgraph).induce s).Connected := by
  /-
    V : Type u
    G : SimpleGraph V
    s : Set V
    ⊢ Iff (SimpleGraph.induce s G).Connected (Top.top.induce s).Connected
  -/
  rw [induce_eq_coe_induce_top, ← Subgraph.connected_iff']
  /-
    🎉 no goals
  -/


lemma induce_union_connected {s t : Set V}
    (sconn : (G.induce s).Connected) (tconn : (G.induce t).Connected)
    (sintert : (s ∩ t).Nonempty) :
    (G.induce (s ∪ t)).Connected := by
  /-
    V : Type u
    G : SimpleGraph V
    s t : Set V
    sconn : (SimpleGraph.induce s G).Connected
    tconn : (SimpleGraph.induce t G).Connected
    sintert : (Inter.inter s t).Nonempty
    ⊢ (SimpleGraph.induce (Union.union s t) G).Connected
  -/
  rw [connected_induce_iff] at sconn tconn ⊢
  /-
    V : Type u
    G : SimpleGraph V
    s t : Set V
    sconn : (Top.top.induce s).Connected
    tconn : (Top.top.induce t).Connected
    sintert : (Inter.inter s t).Nonempty
    ⊢ (Top.top.induce (Union.union s t)).Connected
  -/
  exact Subgraph.induce_union_connected sconn tconn sintert
  /-
    🎉 no goals
  -/


lemma induce_pair_connected_of_adj {u v : V} (huv : G.Adj u v) :
    (G.induce {u, v}).Connected := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    huv : G.Adj u v
    ⊢ (SimpleGraph.induce (Insert.insert u (Singleton.singleton v)) G).Connected
  -/
  rw [connected_induce_iff]
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    huv : G.Adj u v
    ⊢ (Top.top.induce (Insert.insert u (Singleton.singleton v))).Connected
  -/
  exact Subgraph.top_induce_pair_connected_of_adj huv
  /-
    🎉 no goals
  -/


lemma Subgraph.Connected.induce_verts {H : G.Subgraph} (h : H.Connected) :
    (G.induce H.verts).Connected := by
  /-
    V : Type u
    G : SimpleGraph V
    H : G.Subgraph
    h : H.Connected
    ⊢ (SimpleGraph.induce H.verts G).Connected
  -/
  rw [connected_induce_iff]
  /-
    V : Type u
    G : SimpleGraph V
    H : G.Subgraph
    h : H.Connected
    ⊢ (Top.top.induce H.verts).Connected
  -/
  exact h.mono le_induce_top_verts (by exact rfl)
  /-
    🎉 no goals
  -/


lemma Walk.connected_induce_support {u v : V} (p : G.Walk u v) :
    (G.induce {v | v ∈ p.support}).Connected := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    ⊢ (SimpleGraph.induce (setOf fun v_1 => Membership.mem p.support v_1) G).Conne …
  -/
  rw [← p.verts_toSubgraph]
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    ⊢ (SimpleGraph.induce p.toSubgraph.verts G).Connected
  -/
  exact p.toSubgraph_connected.induce_verts
  /-
    🎉 no goals
  -/


lemma induce_connected_adj_union {v w : V} {s t : Set V}
    (sconn : (G.induce s).Connected) (tconn : (G.induce t).Connected)
    (hv : v ∈ s) (hw : w ∈ t) (ha : G.Adj v w) :
    (G.induce (s ∪ t)).Connected := by
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    s t : Set V
    sconn : (SimpleGraph.induce s G).Connected
    tconn : (SimpleGraph.induce t G).Connected
    hv : Membership.mem s v
    hw : Membership.mem t w
    ha : G.Adj v w
    ⊢ (SimpleGraph.induce (Union.union s t) G).Connected
  -/
  rw [connected_induce_iff] at sconn tconn ⊢
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    s t : Set V
    sconn : (Top.top.induce s).Connected
    tconn : (Top.top.induce t).Connected
    hv : Membership.mem s v
    hw : Membership.mem t w
    ha : G.Adj v w
    ⊢ (Top.top.induce (Union.union s t)).Connected
  -/
  apply (sconn.adj_union tconn hv hw ha).mono
  · simp only [Set.mem_singleton_iff, sup_le_iff, Subgraph.le_induce_union_left,
      Subgraph.le_induce_union_right, and_true, ← Subgraph.subgraphOfAdj_eq_induce ha]
    /-
      case hle
      V : Type u
      G : SimpleGraph V
      v w : V
      s t : Set V
      sconn : (Top.top.induce s).Connected
      tconn : (Top.top.induce t).Connected
      hv : Membership.mem s v
      hw : Membership.mem t w
      ha : G.Adj v w
      ⊢ LE.le (G.subgraphOfAdj ha) (Top.top.induce (Union.union s t))
    -/
    apply subgraphOfAdj_le_of_adj
    /-
      case hle.h
      V : Type u
      G : SimpleGraph V
      v w : V
      s t : Set V
      sconn : (Top.top.induce s).Connected
      tconn : (Top.top.induce t).Connected
      hv : Membership.mem s v
      hw : Membership.mem t w
      ha : G.Adj v w
      ⊢ (Top.top.induce (Union.union s t)).Adj v w
    -/
    simp [hv, hw, ha]
    /-
      🎉 no goals
    -/
    /-
      case hv
      V : Type u
      G : SimpleGraph V
      v w : V
      s t : Set V
      sconn : (Top.top.induce s).Connected
      tconn : (Top.top.induce t).Connected
      hv : Membership.mem s v
      hw : Membership.mem t w
      ha : G.Adj v w
      ⊢ Eq (Max.max (Max.max (Top.top.induce (Insert.insert v (Singleton.singleton w …
    -/
  · simp only [Set.mem_singleton_iff, sup_le_iff, Subgraph.verts_sup, Subgraph.induce_verts]
    /-
      case hv
      V : Type u
      G : SimpleGraph V
      v w : V
      s t : Set V
      sconn : (Top.top.induce s).Connected
      tconn : (Top.top.induce t).Connected
      hv : Membership.mem s v
      hw : Membership.mem t w
      ha : G.Adj v w
      ⊢ Eq (Union.union (Union.union (Insert.insert v (Singleton.singleton w)) s) t) …
    -/
    rw [Set.union_assoc]
    /-
      case hv
      V : Type u
      G : SimpleGraph V
      v w : V
      s t : Set V
      sconn : (Top.top.induce s).Connected
      tconn : (Top.top.induce t).Connected
      hv : Membership.mem s v
      hw : Membership.mem t w
      ha : G.Adj v w
      ⊢ Eq (Union.union (Insert.insert v (Singleton.singleton w)) (Union.union s t)) …
    -/
    simp [Set.insert_subset_iff, Set.singleton_subset_iff, hv, hw]
    /-
      🎉 no goals
    -/


lemma induce_connected_of_patches {s : Set V} (u : V) (hu : u ∈ s)
    (patches : ∀ {v}, v ∈ s → ∃ s' ⊆ s, ∃ (hu' : u ∈ s') (hv' : v ∈ s'),
                  (G.induce s').Reachable ⟨u, hu'⟩ ⟨v, hv'⟩) : (G.induce s).Connected := by
  /-
    V : Type u
    G : SimpleGraph V
    s : Set V
    u : V
    hu : Membership.mem s u
    patches : ∀ {v : V}, Membership.mem s v → Exists fun s' => And (HasSubset.Subs …
    ⊢ (SimpleGraph.induce s G).Connected
  -/
  rw [connected_iff_exists_forall_reachable]
  /-
    V : Type u
    G : SimpleGraph V
    s : Set V
    u : V
    hu : Membership.mem s u
    patches : ∀ {v : V}, Membership.mem s v → Exists fun s' => And (HasSubset.Subs …
    ⊢ Exists fun v => ∀ (w : ↑s), (SimpleGraph.induce s G).Reachable v w
  -/
  refine ⟨⟨u, hu⟩, ?_⟩
  /-
    V : Type u
    G : SimpleGraph V
    s : Set V
    u : V
    hu : Membership.mem s u
    patches : ∀ {v : V}, Membership.mem s v → Exists fun s' => And (HasSubset.Subs …
    ⊢ ∀ (w : ↑s), (SimpleGraph.induce s G).Reachable ⟨u, hu⟩ w
  -/
  rintro ⟨v, hv⟩
  /-
    case mk
    V : Type u
    G : SimpleGraph V
    s : Set V
    u : V
    hu : Membership.mem s u
    patches : ∀ {v : V}, Membership.mem s v → Exists fun s' => And (HasSubset.Subs …
    v : V
    hv : Membership.mem s v
    ⊢ (SimpleGraph.induce s G).Reachable ⟨u, hu⟩ ⟨v, hv⟩
  -/
  obtain ⟨sv, svs, hu', hv', uv⟩ := patches hv
  /-
    case mk.intro.intro.intro.intro
    V : Type u
    G : SimpleGraph V
    s : Set V
    u : V
    hu : Membership.mem s u
    patches : ∀ {v : V}, Membership.mem s v → Exists fun s' => And (HasSubset.Subs …
    v : V
    hv : Membership.mem s v
    sv : Set V
    svs : HasSubset.Subset sv s
    hu' : Membership.mem sv u
    hv' : Membership.mem sv v
    uv : (SimpleGraph.induce sv G).Reachable ⟨u, hu'⟩ ⟨v, hv'⟩
    ⊢ (SimpleGraph.induce s G).Reachable ⟨u, hu⟩ ⟨v, hv⟩
  -/
  exact uv.map (induceHomOfLE _ svs).toHom
  /-
    🎉 no goals
  -/


lemma induce_sUnion_connected_of_pairwise_not_disjoint {S : Set (Set V)} (Sn : S.Nonempty)
    (Snd : ∀ {s t}, s ∈ S → t ∈ S → (s ∩ t).Nonempty)
    (Sc : ∀ {s}, s ∈ S → (G.induce s).Connected) :
    (G.induce (⋃₀ S)).Connected := by
  /-
    V : Type u
    G : SimpleGraph V
    S : Set (Set V)
    Sn : S.Nonempty
    Snd : ∀ {s t : Set V}, Membership.mem S s → Membership.mem S t → (Inter.inter  …
    Sc : ∀ {s : Set V}, Membership.mem S s → (SimpleGraph.induce s G).Connected
    ⊢ (SimpleGraph.induce S.sUnion G).Connected
  -/
  obtain ⟨s, sS⟩ := Sn
  /-
    case intro
    V : Type u
    G : SimpleGraph V
    S : Set (Set V)
    Snd : ∀ {s t : Set V}, Membership.mem S s → Membership.mem S t → (Inter.inter  …
    Sc : ∀ {s : Set V}, Membership.mem S s → (SimpleGraph.induce s G).Connected
    s : Set V
    sS : Membership.mem S s
    ⊢ (SimpleGraph.induce S.sUnion G).Connected
  -/
  obtain ⟨v, vs⟩ := (Sc sS).nonempty
  /-
    case intro.intro.mk
    V : Type u
    G : SimpleGraph V
    S : Set (Set V)
    Snd : ∀ {s t : Set V}, Membership.mem S s → Membership.mem S t → (Inter.inter  …
    Sc : ∀ {s : Set V}, Membership.mem S s → (SimpleGraph.induce s G).Connected
    s : Set V
    sS : Membership.mem S s
    v : V
    vs : Membership.mem s v
    ⊢ (SimpleGraph.induce S.sUnion G).Connected
  -/
  apply G.induce_connected_of_patches _ (Set.subset_sUnion_of_mem sS vs)
  /-
    case intro.intro.mk
    V : Type u
    G : SimpleGraph V
    S : Set (Set V)
    Snd : ∀ {s t : Set V}, Membership.mem S s → Membership.mem S t → (Inter.inter  …
    Sc : ∀ {s : Set V}, Membership.mem S s → (SimpleGraph.induce s G).Connected
    s : Set V
    sS : Membership.mem S s
    v : V
    vs : Membership.mem s v
    ⊢ ∀ {v_1 : V}, Membership.mem S.sUnion v_1 → Exists fun s' => And (HasSubset.S …
  -/
  rintro w hw
  /-
    case intro.intro.mk
    V : Type u
    G : SimpleGraph V
    S : Set (Set V)
    Snd : ∀ {s t : Set V}, Membership.mem S s → Membership.mem S t → (Inter.inter  …
    Sc : ∀ {s : Set V}, Membership.mem S s → (SimpleGraph.induce s G).Connected
    s : Set V
    sS : Membership.mem S s
    v : V
    vs : Membership.mem s v
    w : V
    hw : Membership.mem S.sUnion w
    ⊢ Exists fun s' => And (HasSubset.Subset s' S.sUnion) (Exists fun hu' => Exist …
  -/
  simp only [Set.mem_sUnion, exists_prop] at hw
  /-
    case intro.intro.mk
    V : Type u
    G : SimpleGraph V
    S : Set (Set V)
    Snd : ∀ {s t : Set V}, Membership.mem S s → Membership.mem S t → (Inter.inter  …
    Sc : ∀ {s : Set V}, Membership.mem S s → (SimpleGraph.induce s G).Connected
    s : Set V
    sS : Membership.mem S s
    v : V
    vs : Membership.mem s v
    w : V
    hw : Exists fun t => And (Membership.mem S t) (Membership.mem t w)
    ⊢ Exists fun s' => And (HasSubset.Subset s' S.sUnion) (Exists fun hu' => Exist …
  -/
  obtain ⟨t, tS, wt⟩ := hw
  refine ⟨s ∪ t, Set.union_subset (Set.subset_sUnion_of_mem sS) (Set.subset_sUnion_of_mem tS),
          Or.inl vs, Or.inr wt, induce_union_connected (Sc sS) (Sc tS) (Snd sS tS) _ _⟩


lemma extend_finset_to_connected (Gpc : G.Preconnected) {t : Finset V} (tn : t.Nonempty) :
    ∃ (t' : Finset V), t ⊆ t' ∧ (G.induce (t' : Set V)).Connected := by
  classical
  obtain ⟨u, ut⟩ := tn
  refine ⟨t.biUnion (fun v => (Gpc u v).some.support.toFinset), fun v vt => ?_, ?_⟩
  · simp only [Finset.mem_biUnion, List.mem_toFinset, exists_prop]
    exact ⟨v, vt, Walk.end_mem_support _⟩
  · apply G.induce_connected_of_patches u
    · simp only [Finset.coe_biUnion, Finset.mem_coe, List.coe_toFinset, Set.mem_iUnion,
                 Set.mem_setOf_eq, Walk.start_mem_support, exists_prop, and_true]
      exact ⟨u, ut⟩
    intros v hv
    simp only [Finset.mem_coe, Finset.mem_biUnion, List.mem_toFinset, exists_prop] at hv
    obtain ⟨w, wt, hw⟩ := hv
    refine ⟨{x | x ∈ (Gpc u w).some.support}, ?_, ?_⟩
    · simp only [Finset.coe_biUnion, Finset.mem_coe, List.coe_toFinset]
      exact fun x xw => Set.mem_iUnion₂.mpr ⟨w,wt,xw⟩
    · simp only [Set.mem_setOf_eq, Walk.start_mem_support, exists_true_left]
      refine ⟨hw, Walk.connected_induce_support _ _ _⟩


