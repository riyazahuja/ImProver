/-- A subgraph of a `SimpleGraph` is a subset of vertices along with a restriction of the adjacency
relation that is symmetric and is supported by the vertex subset.  They also form a bounded lattice.

Thinking of `V → V → Prop` as `Set (V × V)`, a set of darts (i.e., half-edges), then
`Subgraph.adj_sub` is that the darts of a subgraph are a subset of the darts of `G`. -/
@[ext]
structure Subgraph {V : Type u} (G : SimpleGraph V) where
  verts : Set V
  Adj : V → V → Prop
  adj_sub : ∀ {v w : V}, Adj v w → G.Adj v w
  edge_vert : ∀ {v w : V}, Adj v w → v ∈ verts
  symm : Symmetric Adj := by aesop_graph -- Porting note: Originally `by obviously`


/-- The one-vertex subgraph. -/
@[simps]
protected def singletonSubgraph (G : SimpleGraph V) (v : V) : G.Subgraph where
  verts := {v}
  Adj := ⊥
  adj_sub := False.elim
  edge_vert := False.elim
  symm _ _ := False.elim


/-- The one-edge subgraph. -/
@[simps]
def subgraphOfAdj (G : SimpleGraph V) {v w : V} (hvw : G.Adj v w) : G.Subgraph where
  verts := {v, w}
  Adj a b := s(v, w) = s(a, b)
  adj_sub h := by
    /-
      ι : Sort u_1
      V : Type u
      W : Type v
      G : SimpleGraph V
      v w : V
      hvw : G.Adj v w
      v✝ w✝ : V
      h : (fun a b => Eq (Sym2.mk { fst := v, snd := w }) (Sym2.mk { fst := a, snd : …
      ⊢ G.Adj v✝ w✝
    -/
    rw [← G.mem_edgeSet, ← h]
    /-
      ι : Sort u_1
      V : Type u
      W : Type v
      G : SimpleGraph V
      v w : V
      hvw : G.Adj v w
      v✝ w✝ : V
      h : (fun a b => Eq (Sym2.mk { fst := v, snd := w }) (Sym2.mk { fst := a, snd : …
      ⊢ Membership.mem G.edgeSet (Sym2.mk { fst := v, snd := w })
    -/
    exact hvw
    /-
      🎉 no goals
    -/
  edge_vert {a b} h := by
    /-
      ι : Sort u_1
      V : Type u
      W : Type v
      G : SimpleGraph V
      v w : V
      hvw : G.Adj v w
      a b : V
      h : (fun a b => Eq (Sym2.mk { fst := v, snd := w }) (Sym2.mk { fst := a, snd : …
      ⊢ Membership.mem (Insert.insert v (Singleton.singleton w)) a
    -/
    apply_fun fun e ↦ a ∈ e at h
    /-
      ι : Sort u_1
      V : Type u
      W : Type v
      G : SimpleGraph V
      v w : V
      hvw : G.Adj v w
      a b : V
      h : Eq (Membership.mem (Sym2.mk { fst := v, snd := w }) a) (Membership.mem (Sy …
      ⊢ Membership.mem (Insert.insert v (Singleton.singleton w)) a
    -/
    simp only [Sym2.mem_iff, true_or, eq_iff_iff, iff_true] at h
    /-
      ι : Sort u_1
      V : Type u
      W : Type v
      G : SimpleGraph V
      v w : V
      hvw : G.Adj v w
      a b : V
      h : Or (Eq a v) (Eq a w)
      ⊢ Membership.mem (Insert.insert v (Singleton.singleton w)) a
    -/
    exact h
    /-
      🎉 no goals
    -/


protected theorem loopless (G' : Subgraph G) : Irreflexive G'.Adj :=
  fun v h ↦ G.loopless v (G'.adj_sub h)


theorem adj_comm (G' : Subgraph G) (v w : V) : G'.Adj v w ↔ G'.Adj w v :=
  ⟨fun x ↦ G'.symm x, fun x ↦ G'.symm x⟩


@[symm]
theorem adj_symm (G' : Subgraph G) {u v : V} (h : G'.Adj u v) : G'.Adj v u :=
  G'.symm h


protected theorem Adj.symm {G' : Subgraph G} {u v : V} (h : G'.Adj u v) : G'.Adj v u :=
  G'.symm h


protected theorem Adj.adj_sub {H : G.Subgraph} {u v : V} (h : H.Adj u v) : G.Adj u v :=
  H.adj_sub h


protected theorem Adj.fst_mem {H : G.Subgraph} {u v : V} (h : H.Adj u v) : u ∈ H.verts :=
  H.edge_vert h


protected theorem Adj.snd_mem {H : G.Subgraph} {u v : V} (h : H.Adj u v) : v ∈ H.verts :=
  h.symm.fst_mem


protected theorem Adj.ne {H : G.Subgraph} {u v : V} (h : H.Adj u v) : u ≠ v :=
  h.adj_sub.ne


/-- Coercion from `G' : Subgraph G` to a `SimpleGraph G'.verts`. -/
@[simps]
protected def coe (G' : Subgraph G) : SimpleGraph G'.verts where
  Adj v w := G'.Adj v w
  symm _ _ h := G'.symm h
  loopless v h := loopless G v (G'.adj_sub h)


@[simp]
theorem coe_adj_sub (G' : Subgraph G) (u v : G'.verts) (h : G'.coe.Adj u v) : G.Adj u v :=
  G'.adj_sub h

-- Given `h : H.Adj u v`, then `h.coe : H.coe.Adj ⟨u, _⟩ ⟨v, _⟩`.

protected theorem Adj.coe {H : G.Subgraph} {u v : V} (h : H.Adj u v) :
    H.coe.Adj ⟨u, H.edge_vert h⟩ ⟨v, H.edge_vert h.symm⟩ := h


instance (G : SimpleGraph V) (H : Subgraph G) [DecidableRel H.Adj] : DecidableRel H.coe.Adj :=
  fun a b ↦ ‹DecidableRel H.Adj› _ _


/-- A subgraph is called a *spanning subgraph* if it contains all the vertices of `G`. -/
def IsSpanning (G' : Subgraph G) : Prop :=
  ∀ v : V, v ∈ G'.verts


theorem isSpanning_iff {G' : Subgraph G} : G'.IsSpanning ↔ G'.verts = Set.univ :=
  Set.eq_univ_iff_forall.symm


protected alias ⟨IsSpanning.verts_eq_univ, _⟩ := isSpanning_iff


/-- Coercion from `Subgraph G` to `SimpleGraph V`.  If `G'` is a spanning
subgraph, then `G'.spanningCoe` yields an isomorphic graph.
In general, this adds in all vertices from `V` as isolated vertices. -/
@[simps]
protected def spanningCoe (G' : Subgraph G) : SimpleGraph V where
  Adj := G'.Adj
  symm := G'.symm
  loopless v hv := G.loopless v (G'.adj_sub hv)


@[simp]
theorem Adj.of_spanningCoe {G' : Subgraph G} {u v : G'.verts} (h : G'.spanningCoe.Adj u v) :
    G.Adj u v :=
  G'.adj_sub h


lemma spanningCoe_le (G' : G.Subgraph) : G'.spanningCoe ≤ G := fun _ _ ↦ G'.3


theorem spanningCoe_inj : G₁.spanningCoe = G₂.spanningCoe ↔ G₁.Adj = G₂.Adj := by
  /-
    V : Type u
    G : SimpleGraph V
    G₁ G₂ : G.Subgraph
    ⊢ Iff (Eq G₁.spanningCoe G₂.spanningCoe) (Eq G₁.Adj G₂.Adj)
  -/
  simp [Subgraph.spanningCoe]
  /-
    🎉 no goals
  -/


/-- `spanningCoe` is equivalent to `coe` for a subgraph that `IsSpanning`. -/
@[simps]
def spanningCoeEquivCoeOfSpanning (G' : Subgraph G) (h : G'.IsSpanning) :
    G'.spanningCoe ≃g G'.coe where
  toFun v := ⟨v, h v⟩
  invFun v := v
  left_inv _ := rfl
  right_inv _ := rfl
  map_rel_iff' := Iff.rfl


/-- A subgraph is called an *induced subgraph* if vertices of `G'` are adjacent if
they are adjacent in `G`. -/
def IsInduced (G' : Subgraph G) : Prop :=
  ∀ {v w : V}, v ∈ G'.verts → w ∈ G'.verts → G.Adj v w → G'.Adj v w


/-- `H.support` is the set of vertices that form edges in the subgraph `H`. -/
def support (H : Subgraph G) : Set V := Rel.dom H.Adj


theorem mem_support (H : Subgraph G) {v : V} : v ∈ H.support ↔ ∃ w, H.Adj v w := Iff.rfl


theorem support_subset_verts (H : Subgraph G) : H.support ⊆ H.verts :=
  fun _ ⟨_, h⟩ ↦ H.edge_vert h


/-- `G'.neighborSet v` is the set of vertices adjacent to `v` in `G'`. -/
def neighborSet (G' : Subgraph G) (v : V) : Set V := {w | G'.Adj v w}


theorem neighborSet_subset (G' : Subgraph G) (v : V) : G'.neighborSet v ⊆ G.neighborSet v :=
  fun _ ↦ G'.adj_sub


theorem neighborSet_subset_verts (G' : Subgraph G) (v : V) : G'.neighborSet v ⊆ G'.verts :=
  fun _ h ↦ G'.edge_vert (adj_symm G' h)


@[simp]
theorem mem_neighborSet (G' : Subgraph G) (v w : V) : w ∈ G'.neighborSet v ↔ G'.Adj v w := Iff.rfl


/-- A subgraph as a graph has equivalent neighbor sets. -/
def coeNeighborSetEquiv {G' : Subgraph G} (v : G'.verts) :
    G'.coe.neighborSet v ≃ G'.neighborSet v where
  toFun w := ⟨w, w.2⟩
  invFun w := ⟨⟨w, G'.edge_vert (G'.adj_symm w.2)⟩, w.2⟩
  left_inv _ := rfl
  right_inv _ := rfl


/-- The edge set of `G'` consists of a subset of edges of `G`. -/
def edgeSet (G' : Subgraph G) : Set (Sym2 V) := Sym2.fromRel G'.symm


theorem edgeSet_subset (G' : Subgraph G) : G'.edgeSet ⊆ G.edgeSet :=
  Sym2.ind (fun _ _ ↦ G'.adj_sub)


@[simp]
protected lemma mem_edgeSet {G' : Subgraph G} {v w : V} : s(v, w) ∈ G'.edgeSet ↔ G'.Adj v w := .rfl


@[simp] lemma edgeSet_coe {G' : G.Subgraph} : G'.coe.edgeSet = Sym2.map (↑) ⁻¹' G'.edgeSet := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    ⊢ Eq G'.coe.edgeSet (Set.preimage (Sym2.map Subtype.val) G'.edgeSet)
  -/
  ext e; induction' e using Sym2.ind with a b; simp
                                               /-
                                                 🎉 no goals
                                               -/


lemma image_coe_edgeSet_coe (G' : G.Subgraph) : Sym2.map (↑) '' G'.coe.edgeSet = G'.edgeSet := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    ⊢ Eq (Set.image (Sym2.map Subtype.val) G'.coe.edgeSet) G'.edgeSet
  -/
  rw [edgeSet_coe, Set.image_preimage_eq_iff]
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    ⊢ HasSubset.Subset G'.edgeSet (Set.range (Sym2.map Subtype.val))
  -/
  rintro e he
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    e : Sym2 V
    he : Membership.mem G'.edgeSet e
    ⊢ Membership.mem (Set.range (Sym2.map Subtype.val)) e
  -/
  induction' e using Sym2.ind with a b
  /-
    case h
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    a b : V
    he : Membership.mem G'.edgeSet (Sym2.mk { fst := a, snd := b })
    ⊢ Membership.mem (Set.range (Sym2.map Subtype.val)) (Sym2.mk { fst := a, snd : …
  -/
  rw [Subgraph.mem_edgeSet] at he
  /-
    case h
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    a b : V
    he : G'.Adj a b
    ⊢ Membership.mem (Set.range (Sym2.map Subtype.val)) (Sym2.mk { fst := a, snd : …
  -/
  exact ⟨s(⟨a, edge_vert _ he⟩, ⟨b, edge_vert _ he.symm⟩), Sym2.map_pair_eq ..⟩
  /-
    🎉 no goals
  -/


theorem mem_verts_of_mem_edge {G' : Subgraph G} {e : Sym2 V} {v : V} (he : e ∈ G'.edgeSet)
    (hv : v ∈ e) : v ∈ G'.verts := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    e : Sym2 V
    v : V
    he : Membership.mem G'.edgeSet e
    hv : Membership.mem e v
    ⊢ Membership.mem G'.verts v
  -/
  induction e
  /-
    case h
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    v x✝ y✝ : V
    he : Membership.mem G'.edgeSet (Sym2.mk { fst := x✝, snd := y✝ })
    hv : Membership.mem (Sym2.mk { fst := x✝, snd := y✝ }) v
    ⊢ Membership.mem G'.verts v
  -/
  rcases Sym2.mem_iff.mp hv with (rfl | rfl)
    /-
      case h.inl
      V : Type u
      G : SimpleGraph V
      G' : G.Subgraph
      v y✝ : V
      he : Membership.mem G'.edgeSet (Sym2.mk { fst := v, snd := y✝ })
      hv : Membership.mem (Sym2.mk { fst := v, snd := y✝ }) v
      ⊢ Membership.mem G'.verts v
    -/
  · exact G'.edge_vert he
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      V : Type u
      G : SimpleGraph V
      G' : G.Subgraph
      v x✝ : V
      he : Membership.mem G'.edgeSet (Sym2.mk { fst := x✝, snd := v })
      hv : Membership.mem (Sym2.mk { fst := x✝, snd := v }) v
      ⊢ Membership.mem G'.verts v
    -/
  · exact G'.edge_vert (G'.symm he)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-01")] alias mem_verts_if_mem_edge := mem_verts_of_mem_edge


/-- The `incidenceSet` is the set of edges incident to a given vertex. -/
def incidenceSet (G' : Subgraph G) (v : V) : Set (Sym2 V) := {e ∈ G'.edgeSet | v ∈ e}


theorem incidenceSet_subset_incidenceSet (G' : Subgraph G) (v : V) :
    G'.incidenceSet v ⊆ G.incidenceSet v :=
  fun _ h ↦ ⟨G'.edgeSet_subset h.1, h.2⟩


theorem incidenceSet_subset (G' : Subgraph G) (v : V) : G'.incidenceSet v ⊆ G'.edgeSet :=
  fun _ h ↦ h.1


/-- Give a vertex as an element of the subgraph's vertex type. -/
abbrev vert (G' : Subgraph G) (v : V) (h : v ∈ G'.verts) : G'.verts := ⟨v, h⟩


/--
Create an equal copy of a subgraph (see `copy_eq`) with possibly different definitional equalities.
See Note [range copy pattern].
-/
def copy (G' : Subgraph G) (V'' : Set V) (hV : V'' = G'.verts)
    (adj' : V → V → Prop) (hadj : adj' = G'.Adj) : Subgraph G where
  verts := V''
  Adj := adj'
  adj_sub := hadj.symm ▸ G'.adj_sub
  edge_vert := hV.symm ▸ hadj.symm ▸ G'.edge_vert
  symm := hadj.symm ▸ G'.symm


theorem copy_eq (G' : Subgraph G) (V'' : Set V) (hV : V'' = G'.verts)
    (adj' : V → V → Prop) (hadj : adj' = G'.Adj) : G'.copy V'' hV adj' hadj = G' :=
  Subgraph.ext hV hadj


/-- The union of two subgraphs. -/
instance : Max G.Subgraph where
  max G₁ G₂ :=
    { verts := G₁.verts ∪ G₂.verts
      Adj := G₁.Adj ⊔ G₂.Adj
      adj_sub := fun hab => Or.elim hab (fun h => G₁.adj_sub h) fun h => G₂.adj_sub h
      edge_vert := Or.imp (fun h => G₁.edge_vert h) fun h => G₂.edge_vert h
      symm := fun _ _ => Or.imp G₁.adj_symm G₂.adj_symm }


/-- The intersection of two subgraphs. -/
instance : Min G.Subgraph where
  min G₁ G₂ :=
    { verts := G₁.verts ∩ G₂.verts
      Adj := G₁.Adj ⊓ G₂.Adj
      adj_sub := fun hab => G₁.adj_sub hab.1
      edge_vert := And.imp (fun h => G₁.edge_vert h) fun h => G₂.edge_vert h
      symm := fun _ _ => And.imp G₁.adj_symm G₂.adj_symm }


/-- The `top` subgraph is `G` as a subgraph of itself. -/
instance : Top G.Subgraph where
  top :=
    { verts := Set.univ
      Adj := G.Adj
      adj_sub := id
      edge_vert := @fun v _ _ => Set.mem_univ v
      symm := G.symm }


/-- The `bot` subgraph is the subgraph with no vertices or edges. -/
instance : Bot G.Subgraph where
  bot :=
    { verts := ∅
      Adj := ⊥
      adj_sub := False.elim
      edge_vert := False.elim
      symm := fun _ _ => id }


instance : SupSet G.Subgraph where
  sSup s :=
    { verts := ⋃ G' ∈ s, verts G'
      Adj := fun a b => ∃ G' ∈ s, Adj G' a b
      adj_sub := by
        /-
          ι : Sort u_1
          V : Type u
          W : Type v
          G : SimpleGraph V
          G₁ G₂ : G.Subgraph
          a b : V
          s : Set G.Subgraph
          ⊢ ∀ {v w : V}, (fun a b => Exists fun G' => And (Membership.mem s G') (G'.Adj  …
        -/
        rintro a b ⟨G', -, hab⟩
        /-
          case intro.intro
          ι : Sort u_1
          V : Type u
          W : Type v
          G : SimpleGraph V
          G₁ G₂ : G.Subgraph
          a✝ b✝ : V
          s : Set G.Subgraph
          a b : V
          G' : G.Subgraph
          hab : G'.Adj a b
          ⊢ G.Adj a b
        -/
        exact G'.adj_sub hab
        /-
          🎉 no goals
        -/
      edge_vert := by
        /-
          ι : Sort u_1
          V : Type u
          W : Type v
          G : SimpleGraph V
          G₁ G₂ : G.Subgraph
          a b : V
          s : Set G.Subgraph
          ⊢ ∀ {v w : V}, (fun a b => Exists fun G' => And (Membership.mem s G') (G'.Adj  …
        -/
        rintro a b ⟨G', hG', hab⟩
        /-
          case intro.intro
          ι : Sort u_1
          V : Type u
          W : Type v
          G : SimpleGraph V
          G₁ G₂ : G.Subgraph
          a✝ b✝ : V
          s : Set G.Subgraph
          a b : V
          G' : G.Subgraph
          hG' : Membership.mem s G'
          hab : G'.Adj a b
          ⊢ Membership.mem (Set.iUnion fun G' => Set.iUnion fun h => G'.verts) a
        -/
        exact Set.mem_iUnion₂_of_mem hG' (G'.edge_vert hab)
        /-
          🎉 no goals
        -/
                              /-
                                ι : Sort u_1
                                V : Type u
                                W : Type v
                                G : SimpleGraph V
                                G₁ G₂ : G.Subgraph
                                a✝ b✝ : V
                                s : Set G.Subgraph
                                a b : V
                                h : (fun a b => Exists fun G' => And (Membership.mem s G') (G'.Adj a b)) a b
                                ⊢ (fun a b => Exists fun G' => And (Membership.mem s G') (G'.Adj a b)) b a
                              -/
      symm := fun a b h => by simpa [adj_comm] using h }
                              /-
                                🎉 no goals
                              -/


instance : InfSet G.Subgraph where
  sInf s :=
    { verts := ⋂ G' ∈ s, verts G'
      Adj := fun a b => (∀ ⦃G'⦄, G' ∈ s → Adj G' a b) ∧ G.Adj a b
      adj_sub := And.right
      edge_vert := fun hab => Set.mem_iInter₂_of_mem fun G' hG' => G'.edge_vert <| hab.1 hG'
      symm := fun _ _ => And.imp (forall₂_imp fun _ _ => Adj.symm) G.adj_symm }


@[simp]
theorem sup_adj : (G₁ ⊔ G₂).Adj a b ↔ G₁.Adj a b ∨ G₂.Adj a b :=
  Iff.rfl


@[simp]
theorem inf_adj : (G₁ ⊓ G₂).Adj a b ↔ G₁.Adj a b ∧ G₂.Adj a b :=
  Iff.rfl


@[simp]
theorem top_adj : (⊤ : Subgraph G).Adj a b ↔ G.Adj a b :=
  Iff.rfl


@[simp]
theorem not_bot_adj : ¬ (⊥ : Subgraph G).Adj a b :=
  not_false


@[simp]
theorem verts_sup (G₁ G₂ : G.Subgraph) : (G₁ ⊔ G₂).verts = G₁.verts ∪ G₂.verts :=
  rfl


@[simp]
theorem verts_inf (G₁ G₂ : G.Subgraph) : (G₁ ⊓ G₂).verts = G₁.verts ∩ G₂.verts :=
  rfl


@[simp]
theorem verts_top : (⊤ : G.Subgraph).verts = Set.univ :=
  rfl


@[simp]
theorem verts_bot : (⊥ : G.Subgraph).verts = ∅ :=
  rfl


@[simp]
theorem sSup_adj {s : Set G.Subgraph} : (sSup s).Adj a b ↔ ∃ G ∈ s, Adj G a b :=
  Iff.rfl


@[simp]
theorem sInf_adj {s : Set G.Subgraph} : (sInf s).Adj a b ↔ (∀ G' ∈ s, Adj G' a b) ∧ G.Adj a b :=
  Iff.rfl


@[simp]
theorem iSup_adj {f : ι → G.Subgraph} : (⨆ i, f i).Adj a b ↔ ∃ i, (f i).Adj a b := by
  /-
    ι : Sort u_1
    V : Type u
    G : SimpleGraph V
    a b : V
    f : ι → G.Subgraph
    ⊢ Iff ((iSup fun i => f i).Adj a b) (Exists fun i => (f i).Adj a b)
  -/
  simp [iSup]
  /-
    🎉 no goals
  -/


@[simp]
theorem iInf_adj {f : ι → G.Subgraph} : (⨅ i, f i).Adj a b ↔ (∀ i, (f i).Adj a b) ∧ G.Adj a b := by
  /-
    ι : Sort u_1
    V : Type u
    G : SimpleGraph V
    a b : V
    f : ι → G.Subgraph
    ⊢ Iff ((iInf fun i => f i).Adj a b) (And (∀ (i : ι), (f i).Adj a b) (G.Adj a b))
  -/
  simp [iInf]
  /-
    🎉 no goals
  -/


theorem sInf_adj_of_nonempty {s : Set G.Subgraph} (hs : s.Nonempty) :
    (sInf s).Adj a b ↔ ∀ G' ∈ s, Adj G' a b :=
  sInf_adj.trans <|
    and_iff_left_of_imp <| by
      /-
        V : Type u
        G : SimpleGraph V
        a b : V
        s : Set G.Subgraph
        hs : s.Nonempty
        ⊢ (∀ (G' : G.Subgraph), Membership.mem s G' → G'.Adj a b) → G.Adj a b
      -/
      obtain ⟨G', hG'⟩ := hs
      /-
        case intro
        V : Type u
        G : SimpleGraph V
        a b : V
        s : Set G.Subgraph
        G' : G.Subgraph
        hG' : Membership.mem s G'
        ⊢ (∀ (G' : G.Subgraph), Membership.mem s G' → G'.Adj a b) → G.Adj a b
      -/
      exact fun h => G'.adj_sub (h _ hG')
      /-
        🎉 no goals
      -/


theorem iInf_adj_of_nonempty [Nonempty ι] {f : ι → G.Subgraph} :
    (⨅ i, f i).Adj a b ↔ ∀ i, (f i).Adj a b := by
  /-
    ι : Sort u_1
    V : Type u
    G : SimpleGraph V
    a b : V
    inst✝ : Nonempty ι
    f : ι → G.Subgraph
    ⊢ Iff ((iInf fun i => f i).Adj a b) (∀ (i : ι), (f i).Adj a b)
  -/
  rw [iInf, sInf_adj_of_nonempty (Set.range_nonempty _)]
  /-
    ι : Sort u_1
    V : Type u
    G : SimpleGraph V
    a b : V
    inst✝ : Nonempty ι
    f : ι → G.Subgraph
    ⊢ Iff (∀ (G' : G.Subgraph), Membership.mem (Set.range fun i => f i) G' → G'.Ad …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem verts_sSup (s : Set G.Subgraph) : (sSup s).verts = ⋃ G' ∈ s, verts G' :=
  rfl


@[simp]
theorem verts_sInf (s : Set G.Subgraph) : (sInf s).verts = ⋂ G' ∈ s, verts G' :=
  rfl


@[simp]
                                                                                    /-
                                                                                      ι : Sort u_1
                                                                                      V : Type u
                                                                                      G : SimpleGraph V
                                                                                      f : ι → G.Subgraph
                                                                                      ⊢ Eq (iSup fun i => f i).verts (Set.iUnion fun i => (f i).verts)
                                                                                    -/
theorem verts_iSup {f : ι → G.Subgraph} : (⨆ i, f i).verts = ⋃ i, (f i).verts := by simp [iSup]
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


@[simp]
                                                                                    /-
                                                                                      ι : Sort u_1
                                                                                      V : Type u
                                                                                      G : SimpleGraph V
                                                                                      f : ι → G.Subgraph
                                                                                      ⊢ Eq (iInf fun i => f i).verts (Set.iInter fun i => (f i).verts)
                                                                                    -/
theorem verts_iInf {f : ι → G.Subgraph} : (⨅ i, f i).verts = ⋂ i, (f i).verts := by simp [iInf]
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


@[simp] lemma coe_bot : (⊥ : G.Subgraph).coe = ⊥ := rfl


@[simp] lemma IsInduced.top : (⊤ : G.Subgraph).IsInduced := fun _ _ ↦ id


/-- The graph isomorphism between the top element of `G.subgraph` and `G`. -/
def topIso : (⊤ : G.Subgraph).coe ≃g G where
  toFun := (↑)
  invFun a := ⟨a, Set.mem_univ _⟩
  left_inv _ := Subtype.eta ..
  right_inv _ := rfl
  map_rel_iff' := .rfl


theorem verts_spanningCoe_injective :
    (fun G' : Subgraph G => (G'.verts, G'.spanningCoe)).Injective := by
  /-
    V : Type u
    G : SimpleGraph V
    ⊢ Function.Injective fun G' => { fst := G'.verts, snd := G'.spanningCoe }
  -/
  intro G₁ G₂ h
  /-
    V : Type u
    G : SimpleGraph V
    G₁ G₂ : G.Subgraph
    h : Eq ((fun G' => { fst := G'.verts, snd := G'.spanningCoe }) G₁) ((fun G' => …
    ⊢ Eq G₁ G₂
  -/
  rw [Prod.ext_iff] at h
  /-
    V : Type u
    G : SimpleGraph V
    G₁ G₂ : G.Subgraph
    h : And (Eq ((fun G' => { fst := G'.verts, snd := G'.spanningCoe }) G₁).1 ((fu …
    ⊢ Eq G₁ G₂
  -/
  exact Subgraph.ext h.1 (spanningCoe_inj.1 h.2)
  /-
    🎉 no goals
  -/


/-- For subgraphs `G₁`, `G₂`, `G₁ ≤ G₂` iff `G₁.verts ⊆ G₂.verts` and
`∀ a b, G₁.adj a b → G₂.adj a b`. -/
instance distribLattice : DistribLattice G.Subgraph :=
  { show DistribLattice G.Subgraph from
      verts_spanningCoe_injective.distribLattice _
        (fun _ _ => rfl) fun _ _ => rfl with
    le := fun x y => x.verts ⊆ y.verts ∧ ∀ ⦃v w : V⦄, x.Adj v w → y.Adj v w }


instance : BoundedOrder (Subgraph G) where
  top := ⊤
  bot := ⊥
  le_top x := ⟨Set.subset_univ _, fun _ _ => x.adj_sub⟩
  bot_le _ := ⟨Set.empty_subset _, fun _ _ => False.elim⟩


/-- Note that subgraphs do not form a Boolean algebra, because of `verts`. -/
def completelyDistribLatticeMinimalAxioms : CompletelyDistribLattice.MinimalAxioms G.Subgraph :=
  { Subgraph.distribLattice with
    le := (· ≤ ·)
    sup := (· ⊔ ·)
    inf := (· ⊓ ·)
    top := ⊤
    bot := ⊥
    le_top := fun G' => ⟨Set.subset_univ _, fun _ _ => G'.adj_sub⟩
    bot_le := fun _ => ⟨Set.empty_subset _, fun _ _ => False.elim⟩
    sSup := sSup
    -- Porting note: needed `apply` here to modify elaboration; previously the term itself was fine.
                                   /-
                                     ι : Sort u_1
                                     V : Type u
                                     W : Type v
                                     G : SimpleGraph V
                                     G₁ G₂ : G.Subgraph
                                     a b : V
                                     s : Set G.Subgraph
                                     G' : G.Subgraph
                                     hG' : Membership.mem s G'
                                     ⊢ HasSubset.Subset G'.verts (SupSet.sSup s).verts
                                   -/
    le_sSup := fun s G' hG' => ⟨by apply Set.subset_iUnion₂ G' hG', fun _ _ hab => ⟨G', hG', hab⟩⟩
                                   /-
                                     🎉 no goals
                                   -/
    sSup_le := fun s G' hG' =>
      ⟨Set.iUnion₂_subset fun _ hH => (hG' _ hH).1, by
        /-
          ι : Sort u_1
          V : Type u
          W : Type v
          G : SimpleGraph V
          G₁ G₂ : G.Subgraph
          a b : V
          s : Set G.Subgraph
          G' : G.Subgraph
          hG' : ∀ (b : G.Subgraph), Membership.mem s b → LE.le b G'
          ⊢ ∀ ⦃v w : V⦄, (SupSet.sSup s).Adj v w → G'.Adj v w
        -/
        rintro a b ⟨H, hH, hab⟩
        /-
          case intro.intro
          ι : Sort u_1
          V : Type u
          W : Type v
          G : SimpleGraph V
          G₁ G₂ : G.Subgraph
          a✝ b✝ : V
          s : Set G.Subgraph
          G' : G.Subgraph
          hG' : ∀ (b : G.Subgraph), Membership.mem s b → LE.le b G'
          a b : V
          H : G.Subgraph
          hH : Membership.mem s H
          hab : H.Adj a b
          ⊢ G'.Adj a b
        -/
        exact (hG' _ hH).2 hab⟩
        /-
          🎉 no goals
        -/
    sInf := sInf
    sInf_le := fun _ G' hG' => ⟨Set.iInter₂_subset G' hG', fun _ _ hab => hab.1 hG'⟩
    le_sInf := fun _ G' hG' =>
      ⟨Set.subset_iInter₂ fun _ hH => (hG' _ hH).1, fun _ _ hab =>
        ⟨fun _ hH => (hG' _ hH).2 hab, G'.adj_sub hab⟩⟩
                                              /-
                                                ι : Sort u_1
                                                V : Type u
                                                W : Type v
                                                G : SimpleGraph V
                                                G₁ G₂ : G.Subgraph
                                                a b : V
                                                ι✝ : Type u
                                                κ✝ : ι✝ → Type u
                                                f : (a : ι✝) → κ✝ a → G.Subgraph
                                                ⊢ Eq (iInf fun a => iSup fun b => f a b).verts (iSup fun g => iInf fun a => f  …
                                              -/
    iInf_iSup_eq := fun f => Subgraph.ext (by simpa using iInf_iSup_eq)
                                              /-
                                                🎉 no goals
                                              -/
          /-
            ι : Sort u_1
            V : Type u
            W : Type v
            G : SimpleGraph V
            G₁ G₂ : G.Subgraph
            a b : V
            ι✝ : Type u
            κ✝ : ι✝ → Type u
            f : (a : ι✝) → κ✝ a → G.Subgraph
            ⊢ Eq (iInf fun a => iSup fun b => f a b).Adj (iSup fun g => iInf fun a => f a  …
          -/
      (by ext; simp [Classical.skolem]) }
               /-
                 🎉 no goals
               -/


instance : CompletelyDistribLattice G.Subgraph :=
  .ofMinimalAxioms completelyDistribLatticeMinimalAxioms


@[gcongr] lemma verts_mono {H H' : G.Subgraph} (h : H ≤ H') : H.verts ⊆ H'.verts := h.1

lemma verts_monotone : Monotone (verts : G.Subgraph → Set V) := fun _ _ h ↦ h.1


@[simps]
instance subgraphInhabited : Inhabited (Subgraph G) := ⟨⊥⟩


@[simp]
theorem neighborSet_sup {H H' : G.Subgraph} (v : V) :
    (H ⊔ H').neighborSet v = H.neighborSet v ∪ H'.neighborSet v := rfl


@[simp]
theorem neighborSet_inf {H H' : G.Subgraph} (v : V) :
    (H ⊓ H').neighborSet v = H.neighborSet v ∩ H'.neighborSet v := rfl


@[simp]
theorem neighborSet_top (v : V) : (⊤ : G.Subgraph).neighborSet v = G.neighborSet v := rfl


@[simp]
theorem neighborSet_bot (v : V) : (⊥ : G.Subgraph).neighborSet v = ∅ := rfl


@[simp]
theorem neighborSet_sSup (s : Set G.Subgraph) (v : V) :
    (sSup s).neighborSet v = ⋃ G' ∈ s, neighborSet G' v := by
  /-
    V : Type u
    G : SimpleGraph V
    s : Set G.Subgraph
    v : V
    ⊢ Eq ((SupSet.sSup s).neighborSet v) (Set.iUnion fun G' => Set.iUnion fun h => …
  -/
  ext
  /-
    case h
    V : Type u
    G : SimpleGraph V
    s : Set G.Subgraph
    v x✝ : V
    ⊢ Iff (Membership.mem ((SupSet.sSup s).neighborSet v) x✝) (Membership.mem (Set …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem neighborSet_sInf (s : Set G.Subgraph) (v : V) :
    (sInf s).neighborSet v = (⋂ G' ∈ s, neighborSet G' v) ∩ G.neighborSet v := by
  /-
    V : Type u
    G : SimpleGraph V
    s : Set G.Subgraph
    v : V
    ⊢ Eq ((InfSet.sInf s).neighborSet v) (Inter.inter (Set.iInter fun G' => Set.iI …
  -/
  ext
  /-
    case h
    V : Type u
    G : SimpleGraph V
    s : Set G.Subgraph
    v x✝ : V
    ⊢ Iff (Membership.mem ((InfSet.sInf s).neighborSet v) x✝) (Membership.mem (Int …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem neighborSet_iSup (f : ι → G.Subgraph) (v : V) :
                                                              /-
                                                                ι : Sort u_1
                                                                V : Type u
                                                                G : SimpleGraph V
                                                                f : ι → G.Subgraph
                                                                v : V
                                                                ⊢ Eq ((iSup fun i => f i).neighborSet v) (Set.iUnion fun i => (f i).neighborSe …
                                                              -/
    (⨆ i, f i).neighborSet v = ⋃ i, (f i).neighborSet v := by simp [iSup]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
theorem neighborSet_iInf (f : ι → G.Subgraph) (v : V) :
                                                                                  /-
                                                                                    ι : Sort u_1
                                                                                    V : Type u
                                                                                    G : SimpleGraph V
                                                                                    f : ι → G.Subgraph
                                                                                    v : V
                                                                                    ⊢ Eq ((iInf fun i => f i).neighborSet v) (Inter.inter (Set.iInter fun i => (f  …
                                                                                  -/
    (⨅ i, f i).neighborSet v = (⋂ i, (f i).neighborSet v) ∩ G.neighborSet v := by simp [iInf]
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


@[simp]
theorem edgeSet_top : (⊤ : Subgraph G).edgeSet = G.edgeSet := rfl


@[simp]
theorem edgeSet_bot : (⊥ : Subgraph G).edgeSet = ∅ :=
                          /-
                            V : Type u
                            G : SimpleGraph V
                            ⊢ ∀ (x y : V), Iff (Membership.mem Bot.bot.edgeSet (Sym2.mk { fst := x, snd := …
                          -/
  Set.ext <| Sym2.ind (by simp)
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem edgeSet_inf {H₁ H₂ : Subgraph G} : (H₁ ⊓ H₂).edgeSet = H₁.edgeSet ∩ H₂.edgeSet :=
                          /-
                            V : Type u
                            G : SimpleGraph V
                            H₁ H₂ : G.Subgraph
                            ⊢ ∀ (x y : V), Iff (Membership.mem (Min.min H₁ H₂).edgeSet (Sym2.mk { fst := x …
                          -/
  Set.ext <| Sym2.ind (by simp)
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem edgeSet_sup {H₁ H₂ : Subgraph G} : (H₁ ⊔ H₂).edgeSet = H₁.edgeSet ∪ H₂.edgeSet :=
                          /-
                            V : Type u
                            G : SimpleGraph V
                            H₁ H₂ : G.Subgraph
                            ⊢ ∀ (x y : V), Iff (Membership.mem (Max.max H₁ H₂).edgeSet (Sym2.mk { fst := x …
                          -/
  Set.ext <| Sym2.ind (by simp)
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem edgeSet_sSup (s : Set G.Subgraph) : (sSup s).edgeSet = ⋃ G' ∈ s, edgeSet G' := by
  /-
    V : Type u
    G : SimpleGraph V
    s : Set G.Subgraph
    ⊢ Eq (SupSet.sSup s).edgeSet (Set.iUnion fun G' => Set.iUnion fun h => G'.edge …
  -/
  ext e
  /-
    case h
    V : Type u
    G : SimpleGraph V
    s : Set G.Subgraph
    e : Sym2 V
    ⊢ Iff (Membership.mem (SupSet.sSup s).edgeSet e) (Membership.mem (Set.iUnion f …
  -/
  induction e
  /-
    case h.h
    V : Type u
    G : SimpleGraph V
    s : Set G.Subgraph
    x✝ y✝ : V
    ⊢ Iff (Membership.mem (SupSet.sSup s).edgeSet (Sym2.mk { fst := x✝, snd := y✝  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem edgeSet_sInf (s : Set G.Subgraph) :
    (sInf s).edgeSet = (⋂ G' ∈ s, edgeSet G') ∩ G.edgeSet := by
  /-
    V : Type u
    G : SimpleGraph V
    s : Set G.Subgraph
    ⊢ Eq (InfSet.sInf s).edgeSet (Inter.inter (Set.iInter fun G' => Set.iInter fun …
  -/
  ext e
  /-
    case h
    V : Type u
    G : SimpleGraph V
    s : Set G.Subgraph
    e : Sym2 V
    ⊢ Iff (Membership.mem (InfSet.sInf s).edgeSet e) (Membership.mem (Inter.inter  …
  -/
  induction e
  /-
    case h.h
    V : Type u
    G : SimpleGraph V
    s : Set G.Subgraph
    x✝ y✝ : V
    ⊢ Iff (Membership.mem (InfSet.sInf s).edgeSet (Sym2.mk { fst := x✝, snd := y✝  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem edgeSet_iSup (f : ι → G.Subgraph) :
                                                  /-
                                                    ι : Sort u_1
                                                    V : Type u
                                                    G : SimpleGraph V
                                                    f : ι → G.Subgraph
                                                    ⊢ Eq (iSup fun i => f i).edgeSet (Set.iUnion fun i => (f i).edgeSet)
                                                  -/
    (⨆ i, f i).edgeSet = ⋃ i, (f i).edgeSet := by simp [iSup]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
theorem edgeSet_iInf (f : ι → G.Subgraph) :
    (⨅ i, f i).edgeSet = (⋂ i, (f i).edgeSet) ∩ G.edgeSet := by
  /-
    ι : Sort u_1
    V : Type u
    G : SimpleGraph V
    f : ι → G.Subgraph
    ⊢ Eq (iInf fun i => f i).edgeSet (Inter.inter (Set.iInter fun i => (f i).edgeS …
  -/
  simp [iInf]
  /-
    🎉 no goals
  -/


@[simp]
theorem spanningCoe_top : (⊤ : Subgraph G).spanningCoe = G := rfl


@[simp]
theorem spanningCoe_bot : (⊥ : Subgraph G).spanningCoe = ⊥ := rfl


/-- Turn a subgraph of a `SimpleGraph` into a member of its subgraph type. -/
@[simps]
def _root_.SimpleGraph.toSubgraph (H : SimpleGraph V) (h : H ≤ G) : G.Subgraph where
  verts := Set.univ
  Adj := H.Adj
  adj_sub e := h e
  edge_vert _ := Set.mem_univ _
  symm := H.symm


theorem support_mono {H H' : Subgraph G} (h : H ≤ H') : H.support ⊆ H'.support :=
  Rel.dom_mono h.2


theorem _root_.SimpleGraph.toSubgraph.isSpanning (H : SimpleGraph V) (h : H ≤ G) :
    (toSubgraph H h).IsSpanning :=
  Set.mem_univ


theorem spanningCoe_le_of_le {H H' : Subgraph G} (h : H ≤ H') : H.spanningCoe ≤ H'.spanningCoe :=
  h.2


/-- The top of the `Subgraph G` lattice is equivalent to the graph itself. -/
def topEquiv : (⊤ : Subgraph G).coe ≃g G where
  toFun v := ↑v
  invFun v := ⟨v, trivial⟩
  left_inv _ := rfl
  right_inv _ := rfl
  map_rel_iff' := Iff.rfl


/-- The bottom of the `Subgraph G` lattice is equivalent to the empty graph on the empty
vertex type. -/
def botEquiv : (⊥ : Subgraph G).coe ≃g (⊥ : SimpleGraph Empty) where
  toFun v := v.property.elim
  invFun v := v.elim
  left_inv := fun ⟨_, h⟩ ↦ h.elim
  right_inv v := v.elim
  map_rel_iff' := Iff.rfl


theorem edgeSet_mono {H₁ H₂ : Subgraph G} (h : H₁ ≤ H₂) : H₁.edgeSet ≤ H₂.edgeSet :=
  Sym2.ind h.2


theorem _root_.Disjoint.edgeSet {H₁ H₂ : Subgraph G} (h : Disjoint H₁ H₂) :
    Disjoint H₁.edgeSet H₂.edgeSet :=
                                /-
                                  V : Type u
                                  G : SimpleGraph V
                                  H₁ H₂ : G.Subgraph
                                  h : Disjoint H₁ H₂
                                  ⊢ LE.le (Min.min H₁.edgeSet H₂.edgeSet) Bot.bot
                                -/
  disjoint_iff_inf_le.mpr <| by simpa using edgeSet_mono h.le_bot
                                /-
                                  🎉 no goals
                                -/


/-- Graph homomorphisms induce a covariant function on subgraphs. -/
@[simps]
protected def map (f : G →g G') (H : G.Subgraph) : G'.Subgraph where
  verts := f '' H.verts
  Adj := Relation.Map H.Adj f f
  adj_sub := by
    /-
      ι : Sort u_1
      V : Type u
      W : Type v
      G : SimpleGraph V
      G₁ G₂ : G.Subgraph
      a b : V
      G' : SimpleGraph W
      f✝ f : G.Hom G'
      H : G.Subgraph
      ⊢ ∀ {v w : W}, Relation.Map H.Adj (⇑f) (⇑f) v w → G'.Adj v w
    -/
    rintro _ _ ⟨u, v, h, rfl, rfl⟩
    /-
      case intro.intro.intro.intro
      ι : Sort u_1
      V : Type u
      W : Type v
      G : SimpleGraph V
      G₁ G₂ : G.Subgraph
      a b : V
      G' : SimpleGraph W
      f✝ f : G.Hom G'
      H : G.Subgraph
      u v : V
      h : H.Adj u v
      ⊢ G'.Adj (f u) (f v)
    -/
    exact f.map_rel (H.adj_sub h)
    /-
      🎉 no goals
    -/
  edge_vert := by
    /-
      ι : Sort u_1
      V : Type u
      W : Type v
      G : SimpleGraph V
      G₁ G₂ : G.Subgraph
      a b : V
      G' : SimpleGraph W
      f✝ f : G.Hom G'
      H : G.Subgraph
      ⊢ ∀ {v w : W}, Relation.Map H.Adj (⇑f) (⇑f) v w → Membership.mem (Set.image (⇑ …
    -/
    rintro _ _ ⟨u, v, h, rfl, rfl⟩
    /-
      case intro.intro.intro.intro
      ι : Sort u_1
      V : Type u
      W : Type v
      G : SimpleGraph V
      G₁ G₂ : G.Subgraph
      a b : V
      G' : SimpleGraph W
      f✝ f : G.Hom G'
      H : G.Subgraph
      u v : V
      h : H.Adj u v
      ⊢ Membership.mem (Set.image (⇑f) H.verts) (f u)
    -/
    exact Set.mem_image_of_mem _ (H.edge_vert h)
    /-
      🎉 no goals
    -/
  symm := by
    /-
      ι : Sort u_1
      V : Type u
      W : Type v
      G : SimpleGraph V
      G₁ G₂ : G.Subgraph
      a b : V
      G' : SimpleGraph W
      f✝ f : G.Hom G'
      H : G.Subgraph
      ⊢ Symmetric (Relation.Map H.Adj ⇑f ⇑f)
    -/
    rintro _ _ ⟨u, v, h, rfl, rfl⟩
    /-
      case intro.intro.intro.intro
      ι : Sort u_1
      V : Type u
      W : Type v
      G : SimpleGraph V
      G₁ G₂ : G.Subgraph
      a b : V
      G' : SimpleGraph W
      f✝ f : G.Hom G'
      H : G.Subgraph
      u v : V
      h : H.Adj u v
      ⊢ Relation.Map H.Adj (⇑f) (⇑f) (f v) (f u)
    -/
    exact ⟨v, u, H.symm h, rfl, rfl⟩
    /-
      🎉 no goals
    -/


                                                               /-
                                                                 V : Type u
                                                                 G : SimpleGraph V
                                                                 H : G.Subgraph
                                                                 ⊢ Eq (SimpleGraph.Subgraph.map SimpleGraph.Hom.id H) H
                                                               -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
@[simp] lemma map_id (H : G.Subgraph) : H.map Hom.id = H := by ext <;> simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


lemma map_comp {U : Type*} {G'' : SimpleGraph U} (H : G.Subgraph) (f : G →g G') (g : G' →g G'') :
                                             /-
                                               V : Type u
                                               W : Type v
                                               G : SimpleGraph V
                                               G' : SimpleGraph W
                                               U : Type u_2
                                               G'' : SimpleGraph U
                                               H : G.Subgraph
                                               f : G.Hom G'
                                               g : G'.Hom G''
                                               ⊢ Eq (SimpleGraph.Subgraph.map (g.comp f) H) (SimpleGraph.Subgraph.map g (Simp …
                                             -/
                                                     /-
                                                       🎉 no goals
                                                     -/
    H.map (g.comp f) = (H.map f).map g := by ext <;> simp [Subgraph.map]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[gcongr] lemma map_mono {H₁ H₂ : G.Subgraph} (hH : H₁ ≤ H₂) : H₁.map f ≤ H₂.map f := by
  /-
    V : Type u
    W : Type v
    G : SimpleGraph V
    G' : SimpleGraph W
    f : G.Hom G'
    H₁ H₂ : G.Subgraph
    hH : LE.le H₁ H₂
    ⊢ LE.le (SimpleGraph.Subgraph.map f H₁) (SimpleGraph.Subgraph.map f H₂)
  -/
  constructor
    /-
      case left
      V : Type u
      W : Type v
      G : SimpleGraph V
      G' : SimpleGraph W
      f : G.Hom G'
      H₁ H₂ : G.Subgraph
      hH : LE.le H₁ H₂
      ⊢ HasSubset.Subset (SimpleGraph.Subgraph.map f H₁).verts (SimpleGraph.Subgraph …
    -/
  · intro
    /-
      case left
      V : Type u
      W : Type v
      G : SimpleGraph V
      G' : SimpleGraph W
      f : G.Hom G'
      H₁ H₂ : G.Subgraph
      hH : LE.le H₁ H₂
      a✝ : W
      ⊢ Membership.mem (SimpleGraph.Subgraph.map f H₁).verts a✝ → Membership.mem (Si …
    -/
    simp only [map_verts, Set.mem_image, forall_exists_index, and_imp]
    /-
      case left
      V : Type u
      W : Type v
      G : SimpleGraph V
      G' : SimpleGraph W
      f : G.Hom G'
      H₁ H₂ : G.Subgraph
      hH : LE.le H₁ H₂
      a✝ : W
      ⊢ ∀ (x : V), Membership.mem H₁.verts x → Eq (f x) a✝ → Exists fun x => And (Me …
    -/
    rintro v hv rfl
    /-
      case left
      V : Type u
      W : Type v
      G : SimpleGraph V
      G' : SimpleGraph W
      f : G.Hom G'
      H₁ H₂ : G.Subgraph
      hH : LE.le H₁ H₂
      v : V
      hv : Membership.mem H₁.verts v
      ⊢ Exists fun x => And (Membership.mem H₂.verts x) (Eq (f x) (f v))
    -/
    exact ⟨_, hH.1 hv, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case right
      V : Type u
      W : Type v
      G : SimpleGraph V
      G' : SimpleGraph W
      f : G.Hom G'
      H₁ H₂ : G.Subgraph
      hH : LE.le H₁ H₂
      ⊢ ∀ ⦃v w : W⦄, (SimpleGraph.Subgraph.map f H₁).Adj v w → (SimpleGraph.Subgraph …
    -/
  · rintro _ _ ⟨u, v, ha, rfl, rfl⟩
    /-
      case right.intro.intro.intro.intro
      V : Type u
      W : Type v
      G : SimpleGraph V
      G' : SimpleGraph W
      f : G.Hom G'
      H₁ H₂ : G.Subgraph
      hH : LE.le H₁ H₂
      u v : V
      ha : H₁.Adj u v
      ⊢ (SimpleGraph.Subgraph.map f H₂).Adj (f u) (f v)
    -/
    exact ⟨_, _, hH.2 ha, rfl, rfl⟩
    /-
      🎉 no goals
    -/


lemma map_monotone : Monotone (Subgraph.map f) := fun _ _ ↦ map_mono


theorem map_sup (f : G →g G') (H₁ H₂ : G.Subgraph) : (H₁ ⊔ H₂).map f = H₁.map f ⊔ H₂.map f := by
  /-
    V : Type u
    W : Type v
    G : SimpleGraph V
    G' : SimpleGraph W
    f : G.Hom G'
    H₁ H₂ : G.Subgraph
    ⊢ Eq (SimpleGraph.Subgraph.map f (Max.max H₁ H₂)) (Max.max (SimpleGraph.Subgra …
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp [Set.image_union, map_adj, sup_adj, Relation.Map, or_and_right, exists_or]
          /-
            🎉 no goals
          -/


/-- Graph homomorphisms induce a contravariant function on subgraphs. -/
@[simps]
protected def comap {G' : SimpleGraph W} (f : G →g G') (H : G'.Subgraph) : G.Subgraph where
  verts := f ⁻¹' H.verts
  Adj u v := G.Adj u v ∧ H.Adj (f u) (f v)
  adj_sub h := h.1
  edge_vert h := Set.mem_preimage.1 (H.edge_vert h.2)
  symm _ _ h := ⟨G.symm h.1, H.symm h.2⟩


theorem comap_monotone {G' : SimpleGraph W} (f : G →g G') : Monotone (Subgraph.comap f) := by
  /-
    V : Type u
    W : Type v
    G : SimpleGraph V
    G' : SimpleGraph W
    f : G.Hom G'
    ⊢ Monotone (SimpleGraph.Subgraph.comap f)
  -/
  intro H H' h
  /-
    V : Type u
    W : Type v
    G : SimpleGraph V
    G' : SimpleGraph W
    f : G.Hom G'
    H H' : G'.Subgraph
    h : LE.le H H'
    ⊢ LE.le (SimpleGraph.Subgraph.comap f H) (SimpleGraph.Subgraph.comap f H')
  -/
  constructor
    /-
      case left
      V : Type u
      W : Type v
      G : SimpleGraph V
      G' : SimpleGraph W
      f : G.Hom G'
      H H' : G'.Subgraph
      h : LE.le H H'
      ⊢ HasSubset.Subset (SimpleGraph.Subgraph.comap f H).verts (SimpleGraph.Subgrap …
    -/
  · intro
    /-
      case left
      V : Type u
      W : Type v
      G : SimpleGraph V
      G' : SimpleGraph W
      f : G.Hom G'
      H H' : G'.Subgraph
      h : LE.le H H'
      a✝ : V
      ⊢ Membership.mem (SimpleGraph.Subgraph.comap f H).verts a✝ → Membership.mem (S …
    -/
    simp only [comap_verts, Set.mem_preimage]
    /-
      case left
      V : Type u
      W : Type v
      G : SimpleGraph V
      G' : SimpleGraph W
      f : G.Hom G'
      H H' : G'.Subgraph
      h : LE.le H H'
      a✝ : V
      ⊢ Membership.mem H.verts (f a✝) → Membership.mem H'.verts (f a✝)
    -/
    apply h.1
    /-
      🎉 no goals
    -/
    /-
      case right
      V : Type u
      W : Type v
      G : SimpleGraph V
      G' : SimpleGraph W
      f : G.Hom G'
      H H' : G'.Subgraph
      h : LE.le H H'
      ⊢ ∀ ⦃v w : V⦄, (SimpleGraph.Subgraph.comap f H).Adj v w → (SimpleGraph.Subgrap …
    -/
  · intro v w
    /-
      case right
      V : Type u
      W : Type v
      G : SimpleGraph V
      G' : SimpleGraph W
      f : G.Hom G'
      H H' : G'.Subgraph
      h : LE.le H H'
      v w : V
      ⊢ (SimpleGraph.Subgraph.comap f H).Adj v w → (SimpleGraph.Subgraph.comap f H') …
    -/
    simp +contextual only [comap_adj, and_imp, true_and]
    /-
      case right
      V : Type u
      W : Type v
      G : SimpleGraph V
      G' : SimpleGraph W
      f : G.Hom G'
      H H' : G'.Subgraph
      h : LE.le H H'
      v w : V
      ⊢ G.Adj v w → H.Adj (f v) (f w) → H'.Adj (f v) (f w)
    -/
    intro
    /-
      case right
      V : Type u
      W : Type v
      G : SimpleGraph V
      G' : SimpleGraph W
      f : G.Hom G'
      H H' : G'.Subgraph
      h : LE.le H H'
      v w : V
      a✝ : G.Adj v w
      ⊢ H.Adj (f v) (f w) → H'.Adj (f v) (f w)
    -/
    apply h.2
    /-
      🎉 no goals
    -/


theorem map_le_iff_le_comap {G' : SimpleGraph W} (f : G →g G') (H : G.Subgraph) (H' : G'.Subgraph) :
    H.map f ≤ H' ↔ H ≤ H'.comap f := by
  /-
    V : Type u
    W : Type v
    G : SimpleGraph V
    G' : SimpleGraph W
    f : G.Hom G'
    H : G.Subgraph
    H' : G'.Subgraph
    ⊢ Iff (LE.le (SimpleGraph.Subgraph.map f H) H') (LE.le H (SimpleGraph.Subgraph …
  -/
  refine ⟨fun h ↦ ⟨fun v hv ↦ ?_, fun v w hvw ↦ ?_⟩, fun h ↦ ⟨fun v ↦ ?_, fun v w ↦ ?_⟩⟩
    /-
      case refine_1
      V : Type u
      W : Type v
      G : SimpleGraph V
      G' : SimpleGraph W
      f : G.Hom G'
      H : G.Subgraph
      H' : G'.Subgraph
      h : LE.le (SimpleGraph.Subgraph.map f H) H'
      v : V
      hv : Membership.mem H.verts v
      ⊢ Membership.mem (SimpleGraph.Subgraph.comap f H').verts v
    -/
  · simp only [comap_verts, Set.mem_preimage]
    /-
      case refine_1
      V : Type u
      W : Type v
      G : SimpleGraph V
      G' : SimpleGraph W
      f : G.Hom G'
      H : G.Subgraph
      H' : G'.Subgraph
      h : LE.le (SimpleGraph.Subgraph.map f H) H'
      v : V
      hv : Membership.mem H.verts v
      ⊢ Membership.mem H'.verts (f v)
    -/
    exact h.1 ⟨v, hv, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      V : Type u
      W : Type v
      G : SimpleGraph V
      G' : SimpleGraph W
      f : G.Hom G'
      H : G.Subgraph
      H' : G'.Subgraph
      h : LE.le (SimpleGraph.Subgraph.map f H) H'
      v w : V
      hvw : H.Adj v w
      ⊢ (SimpleGraph.Subgraph.comap f H').Adj v w
    -/
  · simp only [H.adj_sub hvw, comap_adj, true_and]
    /-
      case refine_2
      V : Type u
      W : Type v
      G : SimpleGraph V
      G' : SimpleGraph W
      f : G.Hom G'
      H : G.Subgraph
      H' : G'.Subgraph
      h : LE.le (SimpleGraph.Subgraph.map f H) H'
      v w : V
      hvw : H.Adj v w
      ⊢ H'.Adj (f v) (f w)
    -/
    exact h.2 ⟨v, w, hvw, rfl, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      V : Type u
      W : Type v
      G : SimpleGraph V
      G' : SimpleGraph W
      f : G.Hom G'
      H : G.Subgraph
      H' : G'.Subgraph
      h : LE.le H (SimpleGraph.Subgraph.comap f H')
      v : W
      ⊢ Membership.mem (SimpleGraph.Subgraph.map f H).verts v → Membership.mem H'.ve …
    -/
  · simp only [map_verts, Set.mem_image, forall_exists_index, and_imp]
    /-
      case refine_3
      V : Type u
      W : Type v
      G : SimpleGraph V
      G' : SimpleGraph W
      f : G.Hom G'
      H : G.Subgraph
      H' : G'.Subgraph
      h : LE.le H (SimpleGraph.Subgraph.comap f H')
      v : W
      ⊢ ∀ (x : V), Membership.mem H.verts x → Eq (f x) v → Membership.mem H'.verts v
    -/
    rintro w hw rfl
    /-
      case refine_3
      V : Type u
      W : Type v
      G : SimpleGraph V
      G' : SimpleGraph W
      f : G.Hom G'
      H : G.Subgraph
      H' : G'.Subgraph
      h : LE.le H (SimpleGraph.Subgraph.comap f H')
      w : V
      hw : Membership.mem H.verts w
      ⊢ Membership.mem H'.verts (f w)
    -/
    exact h.1 hw
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      V : Type u
      W : Type v
      G : SimpleGraph V
      G' : SimpleGraph W
      f : G.Hom G'
      H : G.Subgraph
      H' : G'.Subgraph
      h : LE.le H (SimpleGraph.Subgraph.comap f H')
      v w : W
      ⊢ (SimpleGraph.Subgraph.map f H).Adj v w → H'.Adj v w
    -/
  · simp only [Relation.Map, map_adj, forall_exists_index, and_imp]
    /-
      case refine_4
      V : Type u
      W : Type v
      G : SimpleGraph V
      G' : SimpleGraph W
      f : G.Hom G'
      H : G.Subgraph
      H' : G'.Subgraph
      h : LE.le H (SimpleGraph.Subgraph.comap f H')
      v w : W
      ⊢ ∀ (x x_1 : V), H.Adj x x_1 → Eq (f x) v → Eq (f x_1) w → H'.Adj v w
    -/
    rintro u u' hu rfl rfl
    /-
      case refine_4
      V : Type u
      W : Type v
      G : SimpleGraph V
      G' : SimpleGraph W
      f : G.Hom G'
      H : G.Subgraph
      H' : G'.Subgraph
      h : LE.le H (SimpleGraph.Subgraph.comap f H')
      u u' : V
      hu : H.Adj u u'
      ⊢ H'.Adj (f u) (f u')
    -/
    exact (h.2 hu).2
    /-
      🎉 no goals
    -/


/-- Given two subgraphs, one a subgraph of the other, there is an induced injective homomorphism of
the subgraphs as graphs. -/
@[simps]
def inclusion {x y : Subgraph G} (h : x ≤ y) : x.coe →g y.coe where
  toFun v := ⟨↑v, And.left h v.property⟩
  map_rel' hvw := h.2 hvw


theorem inclusion.injective {x y : Subgraph G} (h : x ≤ y) : Function.Injective (inclusion h) := by
  /-
    V : Type u
    G : SimpleGraph V
    x y : G.Subgraph
    h : LE.le x y
    ⊢ Function.Injective ⇑(SimpleGraph.Subgraph.inclusion h)
  -/
  intro v w h
  /-
    V : Type u
    G : SimpleGraph V
    x y : G.Subgraph
    h✝ : LE.le x y
    v w : ↑x.verts
    h : Eq ((SimpleGraph.Subgraph.inclusion h✝) v) ((SimpleGraph.Subgraph.inclusio …
    ⊢ Eq v w
  -/
  rw [inclusion, DFunLike.coe, Subtype.mk_eq_mk] at h
  /-
    V : Type u
    G : SimpleGraph V
    x y : G.Subgraph
    h✝ : LE.le x y
    v w : ↑x.verts
    h : Eq ↑(RelHom.instFunLike.1 { toFun := fun v => ⟨↑v, ⋯⟩, map_rel' := ⋯ } v)  …
    ⊢ Eq v w
  -/
  exact Subtype.ext h
  /-
    🎉 no goals
  -/


/-- There is an induced injective homomorphism of a subgraph of `G` into `G`. -/
@[simps]
protected def hom (x : Subgraph G) : x.coe →g G where
  toFun v := v
  map_rel' := x.adj_sub


@[simp] lemma coe_hom (x : Subgraph G) :
    (x.hom : x.verts → V) = (fun (v : x.verts) => (v : V)) := rfl


theorem hom.injective {x : Subgraph G} : Function.Injective x.hom :=
  fun _ _ ↦ Subtype.ext


/-- There is an induced injective homomorphism of a subgraph of `G` as
a spanning subgraph into `G`. -/
@[simps]
def spanningHom (x : Subgraph G) : x.spanningCoe →g G where
  toFun := id
  map_rel' := x.adj_sub


theorem spanningHom.injective {x : Subgraph G} : Function.Injective x.spanningHom :=
  fun _ _ ↦ id


theorem neighborSet_subset_of_subgraph {x y : Subgraph G} (h : x ≤ y) (v : V) :
    x.neighborSet v ⊆ y.neighborSet v :=
  fun _ h' ↦ h.2 h'


instance neighborSet.decidablePred (G' : Subgraph G) [h : DecidableRel G'.Adj] (v : V) :
    DecidablePred (· ∈ G'.neighborSet v) :=
  h v


/-- If a graph is locally finite at a vertex, then so is a subgraph of that graph. -/
instance finiteAt {G' : Subgraph G} (v : G'.verts) [DecidableRel G'.Adj]
    [Fintype (G.neighborSet v)] : Fintype (G'.neighborSet v) :=
  Set.fintypeSubset (G.neighborSet v) (G'.neighborSet_subset v)


/-- If a subgraph is locally finite at a vertex, then so are subgraphs of that subgraph.

This is not an instance because `G''` cannot be inferred. -/
def finiteAtOfSubgraph {G' G'' : Subgraph G} [DecidableRel G'.Adj] (h : G' ≤ G'') (v : G'.verts)
    [Fintype (G''.neighborSet v)] : Fintype (G'.neighborSet v) :=
  Set.fintypeSubset (G''.neighborSet v) (neighborSet_subset_of_subgraph h v)


instance (G' : Subgraph G) [Fintype G'.verts] (v : V) [DecidablePred (· ∈ G'.neighborSet v)] :
    Fintype (G'.neighborSet v) :=
  Set.fintypeSubset G'.verts (neighborSet_subset_verts G' v)


instance coeFiniteAt {G' : Subgraph G} (v : G'.verts) [Fintype (G'.neighborSet v)] :
    Fintype (G'.coe.neighborSet v) :=
  Fintype.ofEquiv _ (coeNeighborSetEquiv v).symm


theorem IsSpanning.card_verts [Fintype V] {G' : Subgraph G} [Fintype G'.verts] (h : G'.IsSpanning) :
    G'.verts.toFinset.card = Fintype.card V := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝¹ : Fintype V
    G' : G.Subgraph
    inst✝ : Fintype ↑G'.verts
    h : G'.IsSpanning
    ⊢ Eq G'.verts.toFinset.card (Fintype.card V)
  -/
  simp only [isSpanning_iff.1 h, Set.toFinset_univ]
  /-
    V : Type u
    G : SimpleGraph V
    inst✝¹ : Fintype V
    G' : G.Subgraph
    inst✝ : Fintype ↑G'.verts
    h : G'.IsSpanning
    ⊢ Eq Finset.univ.card (Fintype.card V)
  -/
  congr
  /-
    🎉 no goals
  -/


/-- The degree of a vertex in a subgraph. It's zero for vertices outside the subgraph. -/
def degree (G' : Subgraph G) (v : V) [Fintype (G'.neighborSet v)] : ℕ :=
  Fintype.card (G'.neighborSet v)


theorem finset_card_neighborSet_eq_degree {G' : Subgraph G} {v : V} [Fintype (G'.neighborSet v)] :
    (G'.neighborSet v).toFinset.card = G'.degree v := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    v : V
    inst✝ : Fintype ↑(G'.neighborSet v)
    ⊢ Eq (G'.neighborSet v).toFinset.card (G'.degree v)
  -/
  rw [degree, Set.toFinset_card]
  /-
    🎉 no goals
  -/


theorem degree_le (G' : Subgraph G) (v : V) [Fintype (G'.neighborSet v)]
    [Fintype (G.neighborSet v)] : G'.degree v ≤ G.degree v := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    v : V
    inst✝¹ : Fintype ↑(G'.neighborSet v)
    inst✝ : Fintype ↑(G.neighborSet v)
    ⊢ LE.le (G'.degree v) (G.degree v)
  -/
  rw [← card_neighborSet_eq_degree]
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    v : V
    inst✝¹ : Fintype ↑(G'.neighborSet v)
    inst✝ : Fintype ↑(G.neighborSet v)
    ⊢ LE.le (G'.degree v) (Fintype.card ↑(G.neighborSet v))
  -/
  exact Set.card_le_card (G'.neighborSet_subset v)
  /-
    🎉 no goals
  -/


theorem degree_le' (G' G'' : Subgraph G) (h : G' ≤ G'') (v : V) [Fintype (G'.neighborSet v)]
    [Fintype (G''.neighborSet v)] : G'.degree v ≤ G''.degree v :=
  Set.card_le_card (neighborSet_subset_of_subgraph h v)


@[simp]
theorem coe_degree (G' : Subgraph G) (v : G'.verts) [Fintype (G'.coe.neighborSet v)]
    [Fintype (G'.neighborSet v)] : G'.coe.degree v = G'.degree v := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    v : ↑G'.verts
    inst✝¹ : Fintype ↑(G'.coe.neighborSet v)
    inst✝ : Fintype ↑(G'.neighborSet ↑v)
    ⊢ Eq (G'.coe.degree v) (G'.degree ↑v)
  -/
  rw [← card_neighborSet_eq_degree]
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    v : ↑G'.verts
    inst✝¹ : Fintype ↑(G'.coe.neighborSet v)
    inst✝ : Fintype ↑(G'.neighborSet ↑v)
    ⊢ Eq (Fintype.card ↑(G'.coe.neighborSet v)) (G'.degree ↑v)
  -/
  exact Fintype.card_congr (coeNeighborSetEquiv v)
  /-
    🎉 no goals
  -/


@[simp]
theorem degree_spanningCoe {G' : G.Subgraph} (v : V) [Fintype (G'.neighborSet v)]
    [Fintype (G'.spanningCoe.neighborSet v)] : G'.spanningCoe.degree v = G'.degree v := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    v : V
    inst✝¹ : Fintype ↑(G'.neighborSet v)
    inst✝ : Fintype ↑(G'.spanningCoe.neighborSet v)
    ⊢ Eq (G'.spanningCoe.degree v) (G'.degree v)
  -/
  rw [← card_neighborSet_eq_degree, Subgraph.degree]
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    v : V
    inst✝¹ : Fintype ↑(G'.neighborSet v)
    inst✝ : Fintype ↑(G'.spanningCoe.neighborSet v)
    ⊢ Eq (Fintype.card ↑(G'.spanningCoe.neighborSet v)) (Fintype.card ↑(G'.neighbo …
  -/
  congr!
  /-
    🎉 no goals
  -/


theorem degree_eq_one_iff_unique_adj {G' : Subgraph G} {v : V} [Fintype (G'.neighborSet v)] :
    G'.degree v = 1 ↔ ∃! w : V, G'.Adj v w := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    v : V
    inst✝ : Fintype ↑(G'.neighborSet v)
    ⊢ Iff (Eq (G'.degree v) 1) (ExistsUnique fun w => G'.Adj v w)
  -/
  rw [← finset_card_neighborSet_eq_degree, Finset.card_eq_one, Finset.singleton_iff_unique_mem]
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    v : V
    inst✝ : Fintype ↑(G'.neighborSet v)
    ⊢ Iff (ExistsUnique fun a => Membership.mem (G'.neighborSet v).toFinset a) (Ex …
  -/
  simp only [Set.mem_toFinset, mem_neighborSet]
  /-
    🎉 no goals
  -/


instance nonempty_singletonSubgraph_verts (v : V) : Nonempty (G.singletonSubgraph v).verts :=
  ⟨⟨v, Set.mem_singleton v⟩⟩


@[simp]
theorem singletonSubgraph_le_iff (v : V) (H : G.Subgraph) :
    G.singletonSubgraph v ≤ H ↔ v ∈ H.verts := by
  /-
    V : Type u
    G : SimpleGraph V
    v : V
    H : G.Subgraph
    ⊢ Iff (LE.le (G.singletonSubgraph v) H) (Membership.mem H.verts v)
  -/
  refine ⟨fun h ↦ h.1 (Set.mem_singleton v), ?_⟩
  /-
    V : Type u
    G : SimpleGraph V
    v : V
    H : G.Subgraph
    ⊢ Membership.mem H.verts v → LE.le (G.singletonSubgraph v) H
  -/
  intro h
  /-
    V : Type u
    G : SimpleGraph V
    v : V
    H : G.Subgraph
    h : Membership.mem H.verts v
    ⊢ LE.le (G.singletonSubgraph v) H
  -/
  constructor
    /-
      case left
      V : Type u
      G : SimpleGraph V
      v : V
      H : G.Subgraph
      h : Membership.mem H.verts v
      ⊢ HasSubset.Subset (G.singletonSubgraph v).verts H.verts
    -/
  · rwa [singletonSubgraph_verts, Set.singleton_subset_iff]
    /-
      🎉 no goals
    -/
    /-
      case right
      V : Type u
      G : SimpleGraph V
      v : V
      H : G.Subgraph
      h : Membership.mem H.verts v
      ⊢ ∀ ⦃v_1 w : V⦄, (G.singletonSubgraph v).Adj v_1 w → H.Adj v_1 w
    -/
  · exact fun _ _ ↦ False.elim
    /-
      🎉 no goals
    -/


@[simp]
theorem map_singletonSubgraph (f : G →g G') {v : V} :
    Subgraph.map f (G.singletonSubgraph v) = G'.singletonSubgraph (f v) := by
  /-
    V : Type u
    W : Type v
    G : SimpleGraph V
    G' : SimpleGraph W
    f : G.Hom G'
    v : V
    ⊢ Eq (SimpleGraph.Subgraph.map f (G.singletonSubgraph v)) (G'.singletonSubgrap …
  -/
  ext <;> simp only [Relation.Map, Subgraph.map_adj, singletonSubgraph_adj, Pi.bot_apply,
    exists_and_left, and_iff_left_iff_imp, IsEmpty.forall_iff, Subgraph.map_verts,
    singletonSubgraph_verts, Set.image_singleton]
  /-
    case Adj.h.h.a
    V : Type u
    W : Type v
    G : SimpleGraph V
    G' : SimpleGraph W
    f : G.Hom G'
    v : V
    x✝¹ x✝ : W
    ⊢ Bot.bot → Exists fun x => And (Eq (f x) x✝¹) (Exists fun x => Eq (f x) x✝)
  -/
  exact False.elim
  /-
    🎉 no goals
  -/


@[simp]
theorem neighborSet_singletonSubgraph (v w : V) : (G.singletonSubgraph v).neighborSet w = ∅ :=
  rfl


@[simp]
theorem edgeSet_singletonSubgraph (v : V) : (G.singletonSubgraph v).edgeSet = ∅ :=
  Sym2.fromRel_bot


theorem eq_singletonSubgraph_iff_verts_eq (H : G.Subgraph) {v : V} :
    H = G.singletonSubgraph v ↔ H.verts = {v} := by
  /-
    V : Type u
    G : SimpleGraph V
    H : G.Subgraph
    v : V
    ⊢ Iff (Eq H (G.singletonSubgraph v)) (Eq H.verts (Singleton.singleton v))
  -/
  refine ⟨fun h ↦ by rw [h, singletonSubgraph_verts], fun h ↦ ?_⟩
  /-
    V : Type u
    G : SimpleGraph V
    H : G.Subgraph
    v : V
    h : Eq H.verts (Singleton.singleton v)
    ⊢ Eq H (G.singletonSubgraph v)
  -/
  ext
    /-
      case verts.h
      V : Type u
      G : SimpleGraph V
      H : G.Subgraph
      v : V
      h : Eq H.verts (Singleton.singleton v)
      x✝ : V
      ⊢ Iff (Membership.mem H.verts x✝) (Membership.mem (G.singletonSubgraph v).vert …
    -/
  · rw [h, singletonSubgraph_verts]
    /-
      🎉 no goals
    -/
    /-
      case Adj.h.h.a
      V : Type u
      G : SimpleGraph V
      H : G.Subgraph
      v : V
      h : Eq H.verts (Singleton.singleton v)
      x✝¹ x✝ : V
      ⊢ Iff (H.Adj x✝¹ x✝) ((G.singletonSubgraph v).Adj x✝¹ x✝)
    -/
  · simp only [Prop.bot_eq_false, singletonSubgraph_adj, Pi.bot_apply, iff_false]
    /-
      case Adj.h.h.a
      V : Type u
      G : SimpleGraph V
      H : G.Subgraph
      v : V
      h : Eq H.verts (Singleton.singleton v)
      x✝¹ x✝ : V
      ⊢ Not (H.Adj x✝¹ x✝)
    -/
    intro ha
    /-
      case Adj.h.h.a
      V : Type u
      G : SimpleGraph V
      H : G.Subgraph
      v : V
      h : Eq H.verts (Singleton.singleton v)
      x✝¹ x✝ : V
      ha : H.Adj x✝¹ x✝
      ⊢ False
    -/
    have ha1 := ha.fst_mem
    /-
      case Adj.h.h.a
      V : Type u
      G : SimpleGraph V
      H : G.Subgraph
      v : V
      h : Eq H.verts (Singleton.singleton v)
      x✝¹ x✝ : V
      ha : H.Adj x✝¹ x✝
      ha1 : Membership.mem H.verts x✝¹
      ⊢ False
    -/
    have ha2 := ha.snd_mem
    /-
      case Adj.h.h.a
      V : Type u
      G : SimpleGraph V
      H : G.Subgraph
      v : V
      h : Eq H.verts (Singleton.singleton v)
      x✝¹ x✝ : V
      ha : H.Adj x✝¹ x✝
      ha1 : Membership.mem H.verts x✝¹
      ha2 : Membership.mem H.verts x✝
      ⊢ False
    -/
    rw [h, Set.mem_singleton_iff] at ha1 ha2
    /-
      case Adj.h.h.a
      V : Type u
      G : SimpleGraph V
      H : G.Subgraph
      v : V
      h : Eq H.verts (Singleton.singleton v)
      x✝¹ x✝ : V
      ha : H.Adj x✝¹ x✝
      ha1 : Eq x✝¹ v
      ha2 : Eq x✝ v
      ⊢ False
    -/
    subst_vars
    /-
      case Adj.h.h.a
      V : Type u
      G : SimpleGraph V
      H : G.Subgraph
      x✝ : V
      ha : H.Adj x✝ x✝
      h : Eq H.verts (Singleton.singleton x✝)
      ⊢ False
    -/
    exact ha.ne rfl
    /-
      🎉 no goals
    -/


instance nonempty_subgraphOfAdj_verts {v w : V} (hvw : G.Adj v w) :
    Nonempty (G.subgraphOfAdj hvw).verts :=
          /-
            ι : Sort u_1
            V : Type u
            W : Type v
            G : SimpleGraph V
            G' : SimpleGraph W
            v w : V
            hvw : G.Adj v w
            ⊢ Membership.mem (G.subgraphOfAdj hvw).verts v
          -/
  ⟨⟨v, by simp⟩⟩
          /-
            🎉 no goals
          -/


@[simp]
theorem edgeSet_subgraphOfAdj {v w : V} (hvw : G.Adj v w) :
    (G.subgraphOfAdj hvw).edgeSet = {s(v, w)} := by
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    hvw : G.Adj v w
    ⊢ Eq (G.subgraphOfAdj hvw).edgeSet (Singleton.singleton (Sym2.mk { fst := v, s …
  -/
  ext e
  /-
    case h
    V : Type u
    G : SimpleGraph V
    v w : V
    hvw : G.Adj v w
    e : Sym2 V
    ⊢ Iff (Membership.mem (G.subgraphOfAdj hvw).edgeSet e) (Membership.mem (Single …
  -/
  refine e.ind ?_
  simp only [eq_comm, Set.mem_singleton_iff, Subgraph.mem_edgeSet, subgraphOfAdj_adj,
    forall₂_true_iff]


lemma subgraphOfAdj_le_of_adj {v w : V} (H : G.Subgraph) (h : H.Adj v w) :
    G.subgraphOfAdj (H.adj_sub h) ≤ H := by
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    H : G.Subgraph
    h : H.Adj v w
    ⊢ LE.le (G.subgraphOfAdj ⋯) H
  -/
  constructor
    /-
      case left
      V : Type u
      G : SimpleGraph V
      v w : V
      H : G.Subgraph
      h : H.Adj v w
      ⊢ HasSubset.Subset (G.subgraphOfAdj ⋯).verts H.verts
    -/
  · intro x
    /-
      case left
      V : Type u
      G : SimpleGraph V
      v w : V
      H : G.Subgraph
      h : H.Adj v w
      x : V
      ⊢ Membership.mem (G.subgraphOfAdj ⋯).verts x → Membership.mem H.verts x
    -/
                           /-
                             🎉 no goals
                           -/
    rintro (rfl | rfl) <;> simp [H.edge_vert h, H.edge_vert h.symm]
                           /-
                             🎉 no goals
                           -/
    /-
      case right
      V : Type u
      G : SimpleGraph V
      v w : V
      H : G.Subgraph
      h : H.Adj v w
      ⊢ ∀ ⦃v_1 w_1 : V⦄, (G.subgraphOfAdj ⋯).Adj v_1 w_1 → H.Adj v_1 w_1
    -/
  · simp only [subgraphOfAdj_adj, Sym2.eq, Sym2.rel_iff]
    /-
      case right
      V : Type u
      G : SimpleGraph V
      v w : V
      H : G.Subgraph
      h : H.Adj v w
      ⊢ ∀ ⦃v_1 w_1 : V⦄, Or (And (Eq v v_1) (Eq w w_1)) (And (Eq v w_1) (Eq w v_1))  …
    -/
                                             /-
                                               🎉 no goals
                                             -/
    rintro _ _ (⟨rfl, rfl⟩ | ⟨rfl, rfl⟩) <;> simp [h, h.symm]
                                             /-
                                               🎉 no goals
                                             -/


theorem subgraphOfAdj_symm {v w : V} (hvw : G.Adj v w) :
    G.subgraphOfAdj hvw.symm = G.subgraphOfAdj hvw := by
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    hvw : G.Adj v w
    ⊢ Eq (G.subgraphOfAdj ⋯) (G.subgraphOfAdj hvw)
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp [or_comm, and_comm]
          /-
            🎉 no goals
          -/


@[simp]
theorem map_subgraphOfAdj (f : G →g G') {v w : V} (hvw : G.Adj v w) :
    Subgraph.map f (G.subgraphOfAdj hvw) = G'.subgraphOfAdj (f.map_adj hvw) := by
  /-
    V : Type u
    W : Type v
    G : SimpleGraph V
    G' : SimpleGraph W
    f : G.Hom G'
    v w : V
    hvw : G.Adj v w
    ⊢ Eq (SimpleGraph.Subgraph.map f (G.subgraphOfAdj hvw)) (G'.subgraphOfAdj ⋯)
  -/
  ext
  · simp only [Subgraph.map_verts, subgraphOfAdj_verts, Set.mem_image, Set.mem_insert_iff,
      Set.mem_singleton_iff]
    /-
      case verts.h
      V : Type u
      W : Type v
      G : SimpleGraph V
      G' : SimpleGraph W
      f : G.Hom G'
      v w : V
      hvw : G.Adj v w
      x✝ : W
      ⊢ Iff (Exists fun x => And (Or (Eq x v) (Eq x w)) (Eq (f x) x✝)) (Or (Eq x✝ (f …
    -/
    constructor
      /-
        case verts.h.mp
        V : Type u
        W : Type v
        G : SimpleGraph V
        G' : SimpleGraph W
        f : G.Hom G'
        v w : V
        hvw : G.Adj v w
        x✝ : W
        ⊢ (Exists fun x => And (Or (Eq x v) (Eq x w)) (Eq (f x) x✝)) → Or (Eq x✝ (f v) …
      -/
                                     /-
                                       🎉 no goals
                                     -/
    · rintro ⟨u, rfl | rfl, rfl⟩ <;> simp
                                     /-
                                       🎉 no goals
                                     -/
      /-
        case verts.h.mpr
        V : Type u
        W : Type v
        G : SimpleGraph V
        G' : SimpleGraph W
        f : G.Hom G'
        v w : V
        hvw : G.Adj v w
        x✝ : W
        ⊢ Or (Eq x✝ (f v)) (Eq x✝ (f w)) → Exists fun x => And (Or (Eq x v) (Eq x w))  …
      -/
    · rintro (rfl | rfl)
        /-
          case verts.h.mpr.inl
          V : Type u
          W : Type v
          G : SimpleGraph V
          G' : SimpleGraph W
          f : G.Hom G'
          v w : V
          hvw : G.Adj v w
          ⊢ Exists fun x => And (Or (Eq x v) (Eq x w)) (Eq (f x) (f v))
        -/
      · use v
        /-
          case h
          V : Type u
          W : Type v
          G : SimpleGraph V
          G' : SimpleGraph W
          f : G.Hom G'
          v w : V
          hvw : G.Adj v w
          ⊢ And (Or (Eq v v) (Eq v w)) (Eq (f v) (f v))
        -/
        simp
        /-
          🎉 no goals
        -/
        /-
          case verts.h.mpr.inr
          V : Type u
          W : Type v
          G : SimpleGraph V
          G' : SimpleGraph W
          f : G.Hom G'
          v w : V
          hvw : G.Adj v w
          ⊢ Exists fun x => And (Or (Eq x v) (Eq x w)) (Eq (f x) (f w))
        -/
      · use w
        /-
          case h
          V : Type u
          W : Type v
          G : SimpleGraph V
          G' : SimpleGraph W
          f : G.Hom G'
          v w : V
          hvw : G.Adj v w
          ⊢ And (Or (Eq w v) (Eq w w)) (Eq (f w) (f w))
        -/
        simp
        /-
          🎉 no goals
        -/
    /-
      case Adj.h.h.a
      V : Type u
      W : Type v
      G : SimpleGraph V
      G' : SimpleGraph W
      f : G.Hom G'
      v w : V
      hvw : G.Adj v w
      x✝¹ x✝ : W
      ⊢ Iff ((SimpleGraph.Subgraph.map f (G.subgraphOfAdj hvw)).Adj x✝¹ x✝) ((G'.sub …
    -/
  · simp only [Relation.Map, Subgraph.map_adj, subgraphOfAdj_adj, Sym2.eq, Sym2.rel_iff]
    /-
      case Adj.h.h.a
      V : Type u
      W : Type v
      G : SimpleGraph V
      G' : SimpleGraph W
      f : G.Hom G'
      v w : V
      hvw : G.Adj v w
      x✝¹ x✝ : W
      ⊢ Iff (Exists fun a => Exists fun b => And (Or (And (Eq v a) (Eq w b)) (And (E …
    -/
    constructor
      /-
        case Adj.h.h.a.mp
        V : Type u
        W : Type v
        G : SimpleGraph V
        G' : SimpleGraph W
        f : G.Hom G'
        v w : V
        hvw : G.Adj v w
        x✝¹ x✝ : W
        ⊢ (Exists fun a => Exists fun b => And (Or (And (Eq v a) (Eq w b)) (And (Eq v  …
      -/
                                                           /-
                                                             🎉 no goals
                                                           -/
    · rintro ⟨a, b, ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩, rfl, rfl⟩ <;> simp
                                                           /-
                                                             🎉 no goals
                                                           -/
      /-
        case Adj.h.h.a.mpr
        V : Type u
        W : Type v
        G : SimpleGraph V
        G' : SimpleGraph W
        f : G.Hom G'
        v w : V
        hvw : G.Adj v w
        x✝¹ x✝ : W
        ⊢ Or (And (Eq (f v) x✝¹) (Eq (f w) x✝)) (And (Eq (f v) x✝) (Eq (f w) x✝¹)) → E …
      -/
    · rintro (⟨rfl, rfl⟩ | ⟨rfl, rfl⟩)
        /-
          case Adj.h.h.a.mpr.inl.intro
          V : Type u
          W : Type v
          G : SimpleGraph V
          G' : SimpleGraph W
          f : G.Hom G'
          v w : V
          hvw : G.Adj v w
          ⊢ Exists fun a => Exists fun b => And (Or (And (Eq v a) (Eq w b)) (And (Eq v b …
        -/
      · use v, w
        /-
          case h
          V : Type u
          W : Type v
          G : SimpleGraph V
          G' : SimpleGraph W
          f : G.Hom G'
          v w : V
          hvw : G.Adj v w
          ⊢ And (Or (And (Eq v v) (Eq w w)) (And (Eq v w) (Eq w v))) (And (Eq (f v) (f v …
        -/
        simp
        /-
          🎉 no goals
        -/
        /-
          case Adj.h.h.a.mpr.inr.intro
          V : Type u
          W : Type v
          G : SimpleGraph V
          G' : SimpleGraph W
          f : G.Hom G'
          v w : V
          hvw : G.Adj v w
          ⊢ Exists fun a => Exists fun b => And (Or (And (Eq v a) (Eq w b)) (And (Eq v b …
        -/
      · use w, v
        /-
          case h
          V : Type u
          W : Type v
          G : SimpleGraph V
          G' : SimpleGraph W
          f : G.Hom G'
          v w : V
          hvw : G.Adj v w
          ⊢ And (Or (And (Eq v w) (Eq w v)) (And (Eq v v) (Eq w w))) (And (Eq (f w) (f w …
        -/
        simp
        /-
          🎉 no goals
        -/


theorem neighborSet_subgraphOfAdj_subset {u v w : V} (hvw : G.Adj v w) :
    (G.subgraphOfAdj hvw).neighborSet u ⊆ {v, w} :=
  (G.subgraphOfAdj hvw).neighborSet_subset_verts _


@[simp]
theorem neighborSet_fst_subgraphOfAdj {v w : V} (hvw : G.Adj v w) :
    (G.subgraphOfAdj hvw).neighborSet v = {w} := by
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    hvw : G.Adj v w
    ⊢ Eq ((G.subgraphOfAdj hvw).neighborSet v) (Singleton.singleton w)
  -/
  ext u
  /-
    case h
    V : Type u
    G : SimpleGraph V
    v w : V
    hvw : G.Adj v w
    u : V
    ⊢ Iff (Membership.mem ((G.subgraphOfAdj hvw).neighborSet v) u) (Membership.mem …
  -/
  suffices w = u ↔ u = w by simpa [hvw.ne.symm] using this
  /-
    case h
    V : Type u
    G : SimpleGraph V
    v w : V
    hvw : G.Adj v w
    u : V
    ⊢ Iff (Eq w u) (Eq u w)
  -/
  rw [eq_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem neighborSet_snd_subgraphOfAdj {v w : V} (hvw : G.Adj v w) :
    (G.subgraphOfAdj hvw).neighborSet w = {v} := by
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    hvw : G.Adj v w
    ⊢ Eq ((G.subgraphOfAdj hvw).neighborSet w) (Singleton.singleton v)
  -/
  rw [subgraphOfAdj_symm hvw.symm]
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    hvw : G.Adj v w
    ⊢ Eq ((G.subgraphOfAdj ⋯).neighborSet w) (Singleton.singleton v)
  -/
  exact neighborSet_fst_subgraphOfAdj hvw.symm
  /-
    🎉 no goals
  -/


@[simp]
theorem neighborSet_subgraphOfAdj_of_ne_of_ne {u v w : V} (hvw : G.Adj v w) (hv : u ≠ v)
    (hw : u ≠ w) : (G.subgraphOfAdj hvw).neighborSet u = ∅ := by
  /-
    V : Type u
    G : SimpleGraph V
    u v w : V
    hvw : G.Adj v w
    hv : Ne u v
    hw : Ne u w
    ⊢ Eq ((G.subgraphOfAdj hvw).neighborSet u) EmptyCollection.emptyCollection
  -/
  ext
  /-
    case h
    V : Type u
    G : SimpleGraph V
    u v w : V
    hvw : G.Adj v w
    hv : Ne u v
    hw : Ne u w
    x✝ : V
    ⊢ Iff (Membership.mem ((G.subgraphOfAdj hvw).neighborSet u) x✝) (Membership.me …
  -/
  simp [hv.symm, hw.symm]
  /-
    🎉 no goals
  -/


theorem neighborSet_subgraphOfAdj [DecidableEq V] {u v w : V} (hvw : G.Adj v w) :
    (G.subgraphOfAdj hvw).neighborSet u =
    (if u = v then {w} else ∅) ∪ if u = w then {v} else ∅ := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    hvw : G.Adj v w
    ⊢ Eq ((G.subgraphOfAdj hvw).neighborSet u) (Union.union (ite (Eq u v) (Singlet …
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
  split_ifs <;> subst_vars <;> simp [*]
                               /-
                                 🎉 no goals
                               -/


theorem singletonSubgraph_fst_le_subgraphOfAdj {u v : V} {h : G.Adj u v} :
    G.singletonSubgraph u ≤ G.subgraphOfAdj h := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    h : G.Adj u v
    ⊢ LE.le (G.singletonSubgraph u) (G.subgraphOfAdj h)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem singletonSubgraph_snd_le_subgraphOfAdj {u v : V} {h : G.Adj u v} :
    G.singletonSubgraph v ≤ G.subgraphOfAdj h := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    h : G.Adj u v
    ⊢ LE.le (G.singletonSubgraph v) (G.subgraphOfAdj h)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma support_subgraphOfAdj {u v : V} (h : G.Adj u v) :
    (G.subgraphOfAdj h).support = {u , v} := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    h : G.Adj u v
    ⊢ Eq (G.subgraphOfAdj h).support (Insert.insert u (Singleton.singleton v))
  -/
  ext
  /-
    case h
    V : Type u
    G : SimpleGraph V
    u v : V
    h : G.Adj u v
    x✝ : V
    ⊢ Iff (Membership.mem (G.subgraphOfAdj h).support x✝) (Membership.mem (Insert. …
  -/
  rw [Subgraph.mem_support]
  /-
    case h
    V : Type u
    G : SimpleGraph V
    u v : V
    h : G.Adj u v
    x✝ : V
    ⊢ Iff (Exists fun w => (G.subgraphOfAdj h).Adj x✝ w) (Membership.mem (Insert.i …
  -/
  simp only [subgraphOfAdj_adj, Sym2.eq, Sym2.rel_iff', Prod.mk.injEq, Prod.swap_prod_mk]
  /-
    case h
    V : Type u
    G : SimpleGraph V
    u v : V
    h : G.Adj u v
    x✝ : V
    ⊢ Iff (Exists fun w => Or (And (Eq u x✝) (Eq v w)) (And (Eq u w) (Eq v x✝))) ( …
  -/
  refine ⟨?_, fun h ↦ h.elim (fun hl ↦ ⟨v, .inl ⟨hl.symm, rfl⟩⟩) fun hr ↦ ⟨u, .inr ⟨rfl, hr.symm⟩⟩⟩
  /-
    case h
    V : Type u
    G : SimpleGraph V
    u v : V
    h : G.Adj u v
    x✝ : V
    ⊢ (Exists fun w => Or (And (Eq u x✝) (Eq v w)) (And (Eq u w) (Eq v x✝))) → Mem …
  -/
  rintro ⟨_, hw⟩
  /-
    case h.intro
    V : Type u
    G : SimpleGraph V
    u v : V
    h : G.Adj u v
    x✝ w✝ : V
    hw : Or (And (Eq u x✝) (Eq v w✝)) (And (Eq u w✝) (Eq v x✝))
    ⊢ Membership.mem (Insert.insert u (Singleton.singleton v)) x✝
  -/
  exact hw.elim (fun h1 ↦ .inl h1.1.symm) fun hr ↦ .inr hr.2.symm
  /-
    🎉 no goals
  -/


/-- Given a subgraph of a subgraph of `G`, construct a subgraph of `G`. -/
protected abbrev coeSubgraph {G' : G.Subgraph} : G'.coe.Subgraph → G.Subgraph :=
  Subgraph.map G'.hom


/-- Given a subgraph of `G`, restrict it to being a subgraph of another subgraph `G'` by
taking the portion of `G` that intersects `G'`. -/
protected abbrev restrict {G' : G.Subgraph} : G.Subgraph → G'.coe.Subgraph :=
  Subgraph.comap G'.hom


@[simp]
lemma verts_coeSubgraph {G' : Subgraph G} (G'' : Subgraph G'.coe) :
    G''.coeSubgraph.verts = (G''.verts : Set V) := rfl


lemma coeSubgraph_adj {G' : G.Subgraph} (G'' : G'.coe.Subgraph) (v w : V) :
    (G'.coeSubgraph G'').Adj v w ↔
      ∃ (hv : v ∈ G'.verts) (hw : w ∈ G'.verts), G''.Adj ⟨v, hv⟩ ⟨w, hw⟩ := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    G'' : G'.coe.Subgraph
    v w : V
    ⊢ Iff ((SimpleGraph.Subgraph.coeSubgraph G'').Adj v w) (Exists fun hv => Exist …
  -/
  simp [Relation.Map]
  /-
    🎉 no goals
  -/


lemma restrict_adj {G' G'' : G.Subgraph} (v w : G'.verts) :
    (G'.restrict G'').Adj v w ↔ G'.Adj v w ∧ G''.Adj v w := Iff.rfl


theorem restrict_coeSubgraph {G' : G.Subgraph} (G'' : G'.coe.Subgraph) :
    Subgraph.restrict (Subgraph.coeSubgraph G'') = G'' := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    G'' : G'.coe.Subgraph
    ⊢ Eq (SimpleGraph.Subgraph.restrict (SimpleGraph.Subgraph.coeSubgraph G'')) G''
  -/
  ext
    /-
      case verts.h
      V : Type u
      G : SimpleGraph V
      G' : G.Subgraph
      G'' : G'.coe.Subgraph
      x✝ : ↑G'.verts
      ⊢ Iff (Membership.mem (SimpleGraph.Subgraph.restrict (SimpleGraph.Subgraph.coe …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case Adj.h.h.a
      V : Type u
      G : SimpleGraph V
      G' : G.Subgraph
      G'' : G'.coe.Subgraph
      x✝¹ x✝ : ↑G'.verts
      ⊢ Iff ((SimpleGraph.Subgraph.restrict (SimpleGraph.Subgraph.coeSubgraph G'')). …
    -/
  · rw [restrict_adj, coeSubgraph_adj]
    /-
      case Adj.h.h.a
      V : Type u
      G : SimpleGraph V
      G' : G.Subgraph
      G'' : G'.coe.Subgraph
      x✝¹ x✝ : ↑G'.verts
      ⊢ Iff (And (G'.Adj ↑x✝¹ ↑x✝) (Exists fun hv => Exists fun hw => G''.Adj ⟨↑x✝¹, …
    -/
    simpa using G''.adj_sub
    /-
      🎉 no goals
    -/


theorem coeSubgraph_injective (G' : G.Subgraph) :
    Function.Injective (Subgraph.coeSubgraph : G'.coe.Subgraph → G.Subgraph) :=
  Function.LeftInverse.injective restrict_coeSubgraph


lemma coeSubgraph_le {H : G.Subgraph} (H' : H.coe.Subgraph) :
    Subgraph.coeSubgraph H' ≤ H := by
  /-
    V : Type u
    G : SimpleGraph V
    H : G.Subgraph
    H' : H.coe.Subgraph
    ⊢ LE.le (SimpleGraph.Subgraph.coeSubgraph H') H
  -/
  constructor
    /-
      case left
      V : Type u
      G : SimpleGraph V
      H : G.Subgraph
      H' : H.coe.Subgraph
      ⊢ HasSubset.Subset (SimpleGraph.Subgraph.coeSubgraph H').verts H.verts
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case right
      V : Type u
      G : SimpleGraph V
      H : G.Subgraph
      H' : H.coe.Subgraph
      ⊢ ∀ ⦃v w : V⦄, (SimpleGraph.Subgraph.coeSubgraph H').Adj v w → H.Adj v w
    -/
  · rintro v w ⟨_, _, h, rfl, rfl⟩
    /-
      case right.intro.intro.intro.intro
      V : Type u
      G : SimpleGraph V
      H : G.Subgraph
      H' : H.coe.Subgraph
      w✝¹ w✝ : ↑H.verts
      h : H'.Adj w✝¹ w✝
      ⊢ H.Adj (H.hom w✝¹) (H.hom w✝)
    -/
    exact H'.adj_sub h
    /-
      🎉 no goals
    -/


lemma coeSubgraph_restrict_eq {H : G.Subgraph} (H' : G.Subgraph) :
    Subgraph.coeSubgraph (H.restrict H') = H ⊓ H' := by
  /-
    V : Type u
    G : SimpleGraph V
    H H' : G.Subgraph
    ⊢ Eq (SimpleGraph.Subgraph.coeSubgraph (SimpleGraph.Subgraph.restrict H')) (Mi …
  -/
  ext
    /-
      case verts.h
      V : Type u
      G : SimpleGraph V
      H H' : G.Subgraph
      x✝ : V
      ⊢ Iff (Membership.mem (SimpleGraph.Subgraph.coeSubgraph (SimpleGraph.Subgraph. …
    -/
  · simp [and_comm]
    /-
      🎉 no goals
    -/
    /-
      case Adj.h.h.a
      V : Type u
      G : SimpleGraph V
      H H' : G.Subgraph
      x✝¹ x✝ : V
      ⊢ Iff ((SimpleGraph.Subgraph.coeSubgraph (SimpleGraph.Subgraph.restrict H')).A …
    -/
  · simp_rw [coeSubgraph_adj, restrict_adj]
    /-
      case Adj.h.h.a
      V : Type u
      G : SimpleGraph V
      H H' : G.Subgraph
      x✝¹ x✝ : V
      ⊢ Iff (Exists fun h => Exists fun h => And (H.Adj x✝¹ x✝) (H'.Adj x✝¹ x✝)) ((M …
    -/
    simp only [exists_and_left, exists_prop, inf_adj, and_congr_right_iff]
    /-
      case Adj.h.h.a
      V : Type u
      G : SimpleGraph V
      H H' : G.Subgraph
      x✝¹ x✝ : V
      ⊢ H.Adj x✝¹ x✝ → Iff (And (Membership.mem H.verts x✝) (And (Membership.mem H.v …
    -/
    intro h
    /-
      case Adj.h.h.a
      V : Type u
      G : SimpleGraph V
      H H' : G.Subgraph
      x✝¹ x✝ : V
      h : H.Adj x✝¹ x✝
      ⊢ Iff (And (Membership.mem H.verts x✝) (And (Membership.mem H.verts x✝¹) (H'.A …
    -/
    simp [H.edge_vert h, H.edge_vert h.symm]
    /-
      🎉 no goals
    -/


/-- Given a subgraph `G'` and a set of vertex pairs, remove all of the corresponding edges
from its edge set, if present.

See also: `SimpleGraph.deleteEdges`. -/
def deleteEdges (G' : G.Subgraph) (s : Set (Sym2 V)) : G.Subgraph where
  verts := G'.verts
  Adj := G'.Adj \ Sym2.ToRel s
  adj_sub h' := G'.adj_sub h'.1
  edge_vert h' := G'.edge_vert h'.1
                 /-
                   ι : Sort u_1
                   V : Type u
                   W : Type v
                   G : SimpleGraph V
                   G' : G.Subgraph
                   s : Set (Sym2 V)
                   a b : V
                   ⊢ SDiff.sdiff G'.Adj (Sym2.ToRel s) a b → SDiff.sdiff G'.Adj (Sym2.ToRel s) b a
                 -/
  symm a b := by simp [G'.adj_comm, Sym2.eq_swap]
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem deleteEdges_verts : (G'.deleteEdges s).verts = G'.verts :=
  rfl


@[simp]
theorem deleteEdges_adj (v w : V) : (G'.deleteEdges s).Adj v w ↔ G'.Adj v w ∧ ¬s(v, w) ∈ s :=
  Iff.rfl


@[simp]
theorem deleteEdges_deleteEdges (s s' : Set (Sym2 V)) :
    (G'.deleteEdges s).deleteEdges s' = G'.deleteEdges (s ∪ s') := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    s s' : Set (Sym2 V)
    ⊢ Eq ((G'.deleteEdges s).deleteEdges s') (G'.deleteEdges (Union.union s s'))
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp [and_assoc, not_or]
          /-
            🎉 no goals
          -/


@[simp]
theorem deleteEdges_empty_eq : G'.deleteEdges ∅ = G' := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    ⊢ Eq (G'.deleteEdges EmptyCollection.emptyCollection) G'
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp
          /-
            🎉 no goals
          -/


@[simp]
theorem deleteEdges_spanningCoe_eq :
    G'.spanningCoe.deleteEdges s = (G'.deleteEdges s).spanningCoe := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    s : Set (Sym2 V)
    ⊢ Eq (G'.spanningCoe.deleteEdges s) (G'.deleteEdges s).spanningCoe
  -/
  ext
  /-
    case Adj.h.h.a
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    s : Set (Sym2 V)
    x✝¹ x✝ : V
    ⊢ Iff ((G'.spanningCoe.deleteEdges s).Adj x✝¹ x✝) ((G'.deleteEdges s).spanning …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem deleteEdges_coe_eq (s : Set (Sym2 G'.verts)) :
    G'.coe.deleteEdges s = (G'.deleteEdges (Sym2.map (↑) '' s)).coe := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    s : Set (Sym2 ↑G'.verts)
    ⊢ Eq (G'.coe.deleteEdges s) (G'.deleteEdges (Set.image (Sym2.map Subtype.val)  …
  -/
  ext ⟨v, hv⟩ ⟨w, hw⟩
  simp only [SimpleGraph.deleteEdges_adj, coe_adj, deleteEdges_adj, Set.mem_image, not_exists,
    not_and, and_congr_right_iff]
  /-
    case Adj.h.mk.h.mk.a
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    s : Set (Sym2 ↑G'.verts)
    v : V
    hv : Membership.mem G'.verts v
    w : V
    hw : Membership.mem G'.verts w
    ⊢ G'.Adj v w → Iff (Not (Membership.mem s (Sym2.mk { fst := ⟨v, hv⟩, snd := ⟨w …
  -/
  intro
  /-
    case Adj.h.mk.h.mk.a
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    s : Set (Sym2 ↑G'.verts)
    v : V
    hv : Membership.mem G'.verts v
    w : V
    hw : Membership.mem G'.verts w
    a✝ : G'.Adj v w
    ⊢ Iff (Not (Membership.mem s (Sym2.mk { fst := ⟨v, hv⟩, snd := ⟨w, hw⟩ }))) (∀ …
  -/
  constructor
    /-
      case Adj.h.mk.h.mk.a.mp
      V : Type u
      G : SimpleGraph V
      G' : G.Subgraph
      s : Set (Sym2 ↑G'.verts)
      v : V
      hv : Membership.mem G'.verts v
      w : V
      hw : Membership.mem G'.verts w
      a✝ : G'.Adj v w
      ⊢ Not (Membership.mem s (Sym2.mk { fst := ⟨v, hv⟩, snd := ⟨w, hw⟩ })) → ∀ (x : …
    -/
  · intro hs
    /-
      case Adj.h.mk.h.mk.a.mp
      V : Type u
      G : SimpleGraph V
      G' : G.Subgraph
      s : Set (Sym2 ↑G'.verts)
      v : V
      hv : Membership.mem G'.verts v
      w : V
      hw : Membership.mem G'.verts w
      a✝ : G'.Adj v w
      hs : Not (Membership.mem s (Sym2.mk { fst := ⟨v, hv⟩, snd := ⟨w, hw⟩ }))
      ⊢ ∀ (x : Sym2 ↑G'.verts), Membership.mem s x → Not (Eq (Sym2.map Subtype.val x …
    -/
    refine Sym2.ind ?_
    /-
      case Adj.h.mk.h.mk.a.mp
      V : Type u
      G : SimpleGraph V
      G' : G.Subgraph
      s : Set (Sym2 ↑G'.verts)
      v : V
      hv : Membership.mem G'.verts v
      w : V
      hw : Membership.mem G'.verts w
      a✝ : G'.Adj v w
      hs : Not (Membership.mem s (Sym2.mk { fst := ⟨v, hv⟩, snd := ⟨w, hw⟩ }))
      ⊢ ∀ (x y : ↑G'.verts), Membership.mem s (Sym2.mk { fst := x, snd := y }) → Not …
    -/
    rintro ⟨v', hv'⟩ ⟨w', hw'⟩
    /-
      case Adj.h.mk.h.mk.a.mp.mk.mk
      V : Type u
      G : SimpleGraph V
      G' : G.Subgraph
      s : Set (Sym2 ↑G'.verts)
      v : V
      hv : Membership.mem G'.verts v
      w : V
      hw : Membership.mem G'.verts w
      a✝ : G'.Adj v w
      hs : Not (Membership.mem s (Sym2.mk { fst := ⟨v, hv⟩, snd := ⟨w, hw⟩ }))
      v' : V
      hv' : Membership.mem G'.verts v'
      w' : V
      hw' : Membership.mem G'.verts w'
      ⊢ Membership.mem s (Sym2.mk { fst := ⟨v', hv'⟩, snd := ⟨w', hw'⟩ }) → Not (Eq  …
    -/
    simp only [Sym2.map_pair_eq, Sym2.eq]
    /-
      case Adj.h.mk.h.mk.a.mp.mk.mk
      V : Type u
      G : SimpleGraph V
      G' : G.Subgraph
      s : Set (Sym2 ↑G'.verts)
      v : V
      hv : Membership.mem G'.verts v
      w : V
      hw : Membership.mem G'.verts w
      a✝ : G'.Adj v w
      hs : Not (Membership.mem s (Sym2.mk { fst := ⟨v, hv⟩, snd := ⟨w, hw⟩ }))
      v' : V
      hv' : Membership.mem G'.verts v'
      w' : V
      hw' : Membership.mem G'.verts w'
      ⊢ Membership.mem s (Sym2.mk { fst := ⟨v', hv'⟩, snd := ⟨w', hw'⟩ }) → Not (Sym …
    -/
    contrapose!
    /-
      case Adj.h.mk.h.mk.a.mp.mk.mk
      V : Type u
      G : SimpleGraph V
      G' : G.Subgraph
      s : Set (Sym2 ↑G'.verts)
      v : V
      hv : Membership.mem G'.verts v
      w : V
      hw : Membership.mem G'.verts w
      a✝ : G'.Adj v w
      hs : Not (Membership.mem s (Sym2.mk { fst := ⟨v, hv⟩, snd := ⟨w, hw⟩ }))
      v' : V
      hv' : Membership.mem G'.verts v'
      w' : V
      hw' : Membership.mem G'.verts w'
      ⊢ Sym2.Rel V { fst := v', snd := w' } { fst := v, snd := w } → Not (Membership …
    -/
                       /-
                         🎉 no goals
                       -/
    rintro (_ | _) <;> simpa only [Sym2.eq_swap]
                       /-
                         🎉 no goals
                       -/
    /-
      case Adj.h.mk.h.mk.a.mpr
      V : Type u
      G : SimpleGraph V
      G' : G.Subgraph
      s : Set (Sym2 ↑G'.verts)
      v : V
      hv : Membership.mem G'.verts v
      w : V
      hw : Membership.mem G'.verts w
      a✝ : G'.Adj v w
      ⊢ (∀ (x : Sym2 ↑G'.verts), Membership.mem s x → Not (Eq (Sym2.map Subtype.val  …
    -/
  · intro h' hs
    /-
      case Adj.h.mk.h.mk.a.mpr
      V : Type u
      G : SimpleGraph V
      G' : G.Subgraph
      s : Set (Sym2 ↑G'.verts)
      v : V
      hv : Membership.mem G'.verts v
      w : V
      hw : Membership.mem G'.verts w
      a✝ : G'.Adj v w
      h' : ∀ (x : Sym2 ↑G'.verts), Membership.mem s x → Not (Eq (Sym2.map Subtype.va …
      hs : Membership.mem s (Sym2.mk { fst := ⟨v, hv⟩, snd := ⟨w, hw⟩ })
      ⊢ False
    -/
    exact h' _ hs rfl
    /-
      🎉 no goals
    -/


theorem coe_deleteEdges_eq (s : Set (Sym2 V)) :
    (G'.deleteEdges s).coe = G'.coe.deleteEdges (Sym2.map (↑) ⁻¹' s) := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    s : Set (Sym2 V)
    ⊢ Eq (G'.deleteEdges s).coe (G'.coe.deleteEdges (Set.preimage (Sym2.map Subtyp …
  -/
  ext ⟨v, hv⟩ ⟨w, hw⟩
  /-
    case Adj.h.mk.h.mk.a
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    s : Set (Sym2 V)
    v : V
    hv : Membership.mem (G'.deleteEdges s).verts v
    w : V
    hw : Membership.mem (G'.deleteEdges s).verts w
    ⊢ Iff ((G'.deleteEdges s).coe.Adj ⟨v, hv⟩ ⟨w, hw⟩) ((G'.coe.deleteEdges (Set.p …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem deleteEdges_le : G'.deleteEdges s ≤ G' := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    s : Set (Sym2 V)
    ⊢ LE.le (G'.deleteEdges s) G'
  -/
                  /-
                    🎉 no goals
                  -/
  constructor <;> simp +contextual [subset_rfl]
                  /-
                    🎉 no goals
                  -/


theorem deleteEdges_le_of_le {s s' : Set (Sym2 V)} (h : s ⊆ s') :
    G'.deleteEdges s' ≤ G'.deleteEdges s := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    s s' : Set (Sym2 V)
    h : HasSubset.Subset s s'
    ⊢ LE.le (G'.deleteEdges s') (G'.deleteEdges s)
  -/
  constructor <;> simp +contextual only [deleteEdges_verts, deleteEdges_adj,
    true_and, and_imp, subset_rfl]
  /-
    case right
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    s s' : Set (Sym2 V)
    h : HasSubset.Subset s s'
    ⊢ ∀ ⦃v w : V⦄, G'.Adj v w → Not (Membership.mem s' (Sym2.mk { fst := v, snd := …
  -/
  exact fun _ _ _ hs' hs ↦ hs' (h hs)
  /-
    🎉 no goals
  -/


@[simp]
theorem deleteEdges_inter_edgeSet_left_eq :
    G'.deleteEdges (G'.edgeSet ∩ s) = G'.deleteEdges s := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    s : Set (Sym2 V)
    ⊢ Eq (G'.deleteEdges (Inter.inter G'.edgeSet s)) (G'.deleteEdges s)
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp +contextual [imp_false]
          /-
            🎉 no goals
          -/


@[simp]
theorem deleteEdges_inter_edgeSet_right_eq :
    G'.deleteEdges (s ∩ G'.edgeSet) = G'.deleteEdges s := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    s : Set (Sym2 V)
    ⊢ Eq (G'.deleteEdges (Inter.inter s G'.edgeSet)) (G'.deleteEdges s)
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp +contextual [imp_false]
          /-
            🎉 no goals
          -/


theorem coe_deleteEdges_le : (G'.deleteEdges s).coe ≤ (G'.coe : SimpleGraph G'.verts) := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    s : Set (Sym2 V)
    ⊢ LE.le (G'.deleteEdges s).coe G'.coe
  -/
  intro v w
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    s : Set (Sym2 V)
    v w : ↑(G'.deleteEdges s).verts
    ⊢ (G'.deleteEdges s).coe.Adj v w → G'.coe.Adj v w
  -/
  simp +contextual
  /-
    🎉 no goals
  -/


theorem spanningCoe_deleteEdges_le (G' : G.Subgraph) (s : Set (Sym2 V)) :
    (G'.deleteEdges s).spanningCoe ≤ G'.spanningCoe :=
  spanningCoe_le_of_le (deleteEdges_le s)


/-- The induced subgraph of a subgraph. The expectation is that `s ⊆ G'.verts` for the usual
notion of an induced subgraph, but, in general, `s` is taken to be the new vertex set and edges
are induced from the subgraph `G'`. -/
@[simps]
def induce (G' : G.Subgraph) (s : Set V) : G.Subgraph where
  verts := s
  Adj u v := u ∈ s ∧ v ∈ s ∧ G'.Adj u v
  adj_sub h := G'.adj_sub h.2.2
  edge_vert h := h.1
  symm _ _ h := ⟨h.2.1, h.1, G'.symm h.2.2⟩


theorem _root_.SimpleGraph.induce_eq_coe_induce_top (s : Set V) :
    G.induce s = ((⊤ : G.Subgraph).induce s).coe := by
  /-
    V : Type u
    G : SimpleGraph V
    s : Set V
    ⊢ Eq (SimpleGraph.induce s G) (Top.top.induce s).coe
  -/
  ext
  /-
    case Adj.h.h.a
    V : Type u
    G : SimpleGraph V
    s : Set V
    x✝¹ x✝ : ↑s
    ⊢ Iff ((SimpleGraph.induce s G).Adj x✝¹ x✝) ((Top.top.induce s).coe.Adj x✝¹ x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem induce_mono (hg : G' ≤ G'') (hs : s ⊆ s') : G'.induce s ≤ G''.induce s' := by
  /-
    V : Type u
    G : SimpleGraph V
    G' G'' : G.Subgraph
    s s' : Set V
    hg : LE.le G' G''
    hs : HasSubset.Subset s s'
    ⊢ LE.le (G'.induce s) (G''.induce s')
  -/
  constructor
    /-
      case left
      V : Type u
      G : SimpleGraph V
      G' G'' : G.Subgraph
      s s' : Set V
      hg : LE.le G' G''
      hs : HasSubset.Subset s s'
      ⊢ HasSubset.Subset (G'.induce s).verts (G''.induce s').verts
    -/
  · simp [hs]
    /-
      🎉 no goals
    -/
    /-
      case right
      V : Type u
      G : SimpleGraph V
      G' G'' : G.Subgraph
      s s' : Set V
      hg : LE.le G' G''
      hs : HasSubset.Subset s s'
      ⊢ ∀ ⦃v w : V⦄, (G'.induce s).Adj v w → (G''.induce s').Adj v w
    -/
  · simp +contextual only [induce_adj, and_imp]
    /-
      case right
      V : Type u
      G : SimpleGraph V
      G' G'' : G.Subgraph
      s s' : Set V
      hg : LE.le G' G''
      hs : HasSubset.Subset s s'
      ⊢ ∀ ⦃v w : V⦄, Membership.mem s v → Membership.mem s w → G'.Adj v w → And (Mem …
    -/
    intro v w hv hw ha
    /-
      case right
      V : Type u
      G : SimpleGraph V
      G' G'' : G.Subgraph
      s s' : Set V
      hg : LE.le G' G''
      hs : HasSubset.Subset s s'
      v w : V
      hv : Membership.mem s v
      hw : Membership.mem s w
      ha : G'.Adj v w
      ⊢ And (Membership.mem s' v) (And (Membership.mem s' w) (G''.Adj v w))
    -/
    exact ⟨hs hv, hs hw, hg.2 ha⟩
    /-
      🎉 no goals
    -/


@[gcongr, mono]
theorem induce_mono_left (hg : G' ≤ G'') : G'.induce s ≤ G''.induce s :=
  induce_mono hg subset_rfl


@[gcongr, mono]
theorem induce_mono_right (hs : s ⊆ s') : G'.induce s ≤ G'.induce s' :=
  induce_mono le_rfl hs


@[simp]
theorem induce_empty : G'.induce ∅ = ⊥ := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    ⊢ Eq (G'.induce EmptyCollection.emptyCollection) Bot.bot
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp
          /-
            🎉 no goals
          -/


@[simp]
theorem induce_self_verts : G'.induce G'.verts = G' := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    ⊢ Eq (G'.induce G'.verts) G'
  -/
  ext
    /-
      case verts.h
      V : Type u
      G : SimpleGraph V
      G' : G.Subgraph
      x✝ : V
      ⊢ Iff (Membership.mem (G'.induce G'.verts).verts x✝) (Membership.mem G'.verts  …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case Adj.h.h.a
      V : Type u
      G : SimpleGraph V
      G' : G.Subgraph
      x✝¹ x✝ : V
      ⊢ Iff ((G'.induce G'.verts).Adj x✝¹ x✝) (G'.Adj x✝¹ x✝)
    -/
  · constructor <;>
      /-
        case Adj.h.h.a.mp
        V : Type u
        G : SimpleGraph V
        G' : G.Subgraph
        x✝¹ x✝ : V
        ⊢ (G'.induce G'.verts).Adj x✝¹ x✝ → G'.Adj x✝¹ x✝
      -/
      /-
        🎉 no goals
      -/
      simp +contextual only [induce_adj, imp_true_iff, and_true]
    /-
      case Adj.h.h.a.mpr
      V : Type u
      G : SimpleGraph V
      G' : G.Subgraph
      x✝¹ x✝ : V
      ⊢ G'.Adj x✝¹ x✝ → And (Membership.mem G'.verts x✝¹) (Membership.mem G'.verts x✝)
    -/
    exact fun ha ↦ ⟨G'.edge_vert ha, G'.edge_vert ha.symm⟩
    /-
      🎉 no goals
    -/


lemma le_induce_top_verts : G' ≤ (⊤ : G.Subgraph).induce G'.verts :=
  calc G' = G'.induce G'.verts               := Subgraph.induce_self_verts.symm
       _  ≤ (⊤ : G.Subgraph).induce G'.verts := Subgraph.induce_mono_left le_top


lemma le_induce_union : G'.induce s ⊔ G'.induce s' ≤ G'.induce (s ∪ s') := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    s s' : Set V
    ⊢ LE.le (Max.max (G'.induce s) (G'.induce s')) (G'.induce (Union.union s s'))
  -/
  constructor
    /-
      case left
      V : Type u
      G : SimpleGraph V
      G' : G.Subgraph
      s s' : Set V
      ⊢ HasSubset.Subset (Max.max (G'.induce s) (G'.induce s')).verts (G'.induce (Un …
    -/
  · simp only [verts_sup, induce_verts, Set.Subset.rfl]
    /-
      🎉 no goals
    -/
    /-
      case right
      V : Type u
      G : SimpleGraph V
      G' : G.Subgraph
      s s' : Set V
      ⊢ ∀ ⦃v w : V⦄, (Max.max (G'.induce s) (G'.induce s')).Adj v w → (G'.induce (Un …
    -/
  · simp only [sup_adj, induce_adj, Set.mem_union]
    /-
      case right
      V : Type u
      G : SimpleGraph V
      G' : G.Subgraph
      s s' : Set V
      ⊢ ∀ ⦃v w : V⦄, Or (And (Membership.mem s v) (And (Membership.mem s w) (G'.Adj  …
    -/
                           /-
                             🎉 no goals
                           -/
    rintro v w (h | h) <;> simp [h]
                           /-
                             🎉 no goals
                           -/


lemma le_induce_union_left : G'.induce s ≤ G'.induce (s ∪ s') := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    s s' : Set V
    ⊢ LE.le (G'.induce s) (G'.induce (Union.union s s'))
  -/
  exact (sup_le_iff.mp le_induce_union).1
  /-
    🎉 no goals
  -/


lemma le_induce_union_right : G'.induce s' ≤ G'.induce (s ∪ s') := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    s s' : Set V
    ⊢ LE.le (G'.induce s') (G'.induce (Union.union s s'))
  -/
  exact (sup_le_iff.mp le_induce_union).2
  /-
    🎉 no goals
  -/


theorem singletonSubgraph_eq_induce {v : V} :
    G.singletonSubgraph v = (⊤ : G.Subgraph).induce {v} := by
  /-
    V : Type u
    G : SimpleGraph V
    v : V
    ⊢ Eq (G.singletonSubgraph v) (Top.top.induce (Singleton.singleton v))
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp +contextual [-Set.bot_eq_empty, Prop.bot_eq_false]
          /-
            🎉 no goals
          -/


theorem subgraphOfAdj_eq_induce {v w : V} (hvw : G.Adj v w) :
    G.subgraphOfAdj hvw = (⊤ : G.Subgraph).induce {v, w} := by
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    hvw : G.Adj v w
    ⊢ Eq (G.subgraphOfAdj hvw) (Top.top.induce (Insert.insert v (Singleton.singlet …
  -/
  ext
    /-
      case verts.h
      V : Type u
      G : SimpleGraph V
      v w : V
      hvw : G.Adj v w
      x✝ : V
      ⊢ Iff (Membership.mem (G.subgraphOfAdj hvw).verts x✝) (Membership.mem (Top.top …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case Adj.h.h.a
      V : Type u
      G : SimpleGraph V
      v w : V
      hvw : G.Adj v w
      x✝¹ x✝ : V
      ⊢ Iff ((G.subgraphOfAdj hvw).Adj x✝¹ x✝) ((Top.top.induce (Insert.insert v (Si …
    -/
  · constructor
      /-
        case Adj.h.h.a.mp
        V : Type u
        G : SimpleGraph V
        v w : V
        hvw : G.Adj v w
        x✝¹ x✝ : V
        ⊢ (G.subgraphOfAdj hvw).Adj x✝¹ x✝ → (Top.top.induce (Insert.insert v (Singlet …
      -/
    · intro h
      /-
        case Adj.h.h.a.mp
        V : Type u
        G : SimpleGraph V
        v w : V
        hvw : G.Adj v w
        x✝¹ x✝ : V
        h : (G.subgraphOfAdj hvw).Adj x✝¹ x✝
        ⊢ (Top.top.induce (Insert.insert v (Singleton.singleton w))).Adj x✝¹ x✝
      -/
      simp only [subgraphOfAdj_adj, Sym2.eq, Sym2.rel_iff] at h
      /-
        case Adj.h.h.a.mp
        V : Type u
        G : SimpleGraph V
        v w : V
        hvw : G.Adj v w
        x✝¹ x✝ : V
        h : Or (And (Eq v x✝¹) (Eq w x✝)) (And (Eq v x✝) (Eq w x✝¹))
        ⊢ (Top.top.induce (Insert.insert v (Singleton.singleton w))).Adj x✝¹ x✝
      -/
                                              /-
                                                🎉 no goals
                                              -/
      obtain ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ := h <;> simp [hvw, hvw.symm]
                                              /-
                                                🎉 no goals
                                              -/
      /-
        case Adj.h.h.a.mpr
        V : Type u
        G : SimpleGraph V
        v w : V
        hvw : G.Adj v w
        x✝¹ x✝ : V
        ⊢ (Top.top.induce (Insert.insert v (Singleton.singleton w))).Adj x✝¹ x✝ → (G.s …
      -/
    · intro h
      /-
        case Adj.h.h.a.mpr
        V : Type u
        G : SimpleGraph V
        v w : V
        hvw : G.Adj v w
        x✝¹ x✝ : V
        h : (Top.top.induce (Insert.insert v (Singleton.singleton w))).Adj x✝¹ x✝
        ⊢ (G.subgraphOfAdj hvw).Adj x✝¹ x✝
      -/
      simp only [induce_adj, Set.mem_insert_iff, Set.mem_singleton_iff, top_adj] at h
      /-
        case Adj.h.h.a.mpr
        V : Type u
        G : SimpleGraph V
        v w : V
        hvw : G.Adj v w
        x✝¹ x✝ : V
        h : And (Or (Eq x✝¹ v) (Eq x✝¹ w)) (And (Or (Eq x✝ v) (Eq x✝ w)) (G.Adj x✝¹ x✝))
        ⊢ (G.subgraphOfAdj hvw).Adj x✝¹ x✝
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
      obtain ⟨rfl | rfl, rfl | rfl, ha⟩ := h <;> first |exact (ha.ne rfl).elim|simp
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- Given a subgraph and a set of vertices, delete all the vertices from the subgraph,
if present. Any edges incident to the deleted vertices are deleted as well. -/
abbrev deleteVerts (G' : G.Subgraph) (s : Set V) : G.Subgraph :=
  G'.induce (G'.verts \ s)


theorem deleteVerts_verts : (G'.deleteVerts s).verts = G'.verts \ s :=
  rfl


theorem deleteVerts_adj {u v : V} :
    (G'.deleteVerts s).Adj u v ↔ u ∈ G'.verts ∧ ¬u ∈ s ∧ v ∈ G'.verts ∧ ¬v ∈ s ∧ G'.Adj u v := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    s : Set V
    u v : V
    ⊢ Iff ((G'.deleteVerts s).Adj u v) (And (Membership.mem G'.verts u) (And (Not  …
  -/
  simp [and_assoc]
  /-
    🎉 no goals
  -/


@[simp]
theorem deleteVerts_deleteVerts (s s' : Set V) :
    (G'.deleteVerts s).deleteVerts s' = G'.deleteVerts (s ∪ s') := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    s s' : Set V
    ⊢ Eq ((G'.deleteVerts s).deleteVerts s') (G'.deleteVerts (Union.union s s'))
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp +contextual [not_or, and_assoc]
          /-
            🎉 no goals
          -/


@[simp]
theorem deleteVerts_empty : G'.deleteVerts ∅ = G' := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    ⊢ Eq (G'.deleteVerts EmptyCollection.emptyCollection) G'
  -/
  simp [deleteVerts]
  /-
    🎉 no goals
  -/


theorem deleteVerts_le : G'.deleteVerts s ≤ G' := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    s : Set V
    ⊢ LE.le (G'.deleteVerts s) G'
  -/
                  /-
                    🎉 no goals
                  -/
  constructor <;> simp [Set.diff_subset]
                  /-
                    🎉 no goals
                  -/


@[gcongr, mono]
theorem deleteVerts_mono {G' G'' : G.Subgraph} (h : G' ≤ G'') :
    G'.deleteVerts s ≤ G''.deleteVerts s :=
  induce_mono h (Set.diff_subset_diff_left h.1)


@[gcongr, mono]
theorem deleteVerts_anti {s s' : Set V} (h : s ⊆ s') : G'.deleteVerts s' ≤ G'.deleteVerts s :=
  induce_mono (le_refl _) (Set.diff_subset_diff_right h)


@[simp]
theorem deleteVerts_inter_verts_left_eq : G'.deleteVerts (G'.verts ∩ s) = G'.deleteVerts s := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    s : Set V
    ⊢ Eq (G'.deleteVerts (Inter.inter G'.verts s)) (G'.deleteVerts s)
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp +contextual [imp_false]
          /-
            🎉 no goals
          -/


@[simp]
theorem deleteVerts_inter_verts_set_right_eq :
    G'.deleteVerts (s ∩ G'.verts) = G'.deleteVerts s := by
  /-
    V : Type u
    G : SimpleGraph V
    G' : G.Subgraph
    s : Set V
    ⊢ Eq (G'.deleteVerts (Inter.inter s G'.verts)) (G'.deleteVerts s)
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp +contextual [imp_false]
          /-
            🎉 no goals
          -/


