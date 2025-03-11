/-- Given an injective function, there is a covariant induced map on graphs by pushing forward
the adjacency relation.

This is injective (see `SimpleGraph.map_injective`). -/
protected def map (f : V ↪ W) (G : SimpleGraph V) : SimpleGraph W where
  Adj := Relation.Map G.Adj f f
  symm a b := by -- Porting note: `obviously` used to handle this
    /-
      V : Type u_1
      W : Type u_2
      X : Type u_3
      G✝ : SimpleGraph V
      G' : SimpleGraph W
      u v : V
      f : Function.Embedding V W
      G : SimpleGraph V
      a b : W
      ⊢ Relation.Map G.Adj (⇑f) (⇑f) a b → Relation.Map G.Adj (⇑f) (⇑f) b a
    -/
    rintro ⟨v, w, h, rfl, rfl⟩
    /-
      case intro.intro.intro.intro
      V : Type u_1
      W : Type u_2
      X : Type u_3
      G✝ : SimpleGraph V
      G' : SimpleGraph W
      u v✝ : V
      f : Function.Embedding V W
      G : SimpleGraph V
      v w : V
      h : G.Adj v w
      ⊢ Relation.Map G.Adj (⇑f) (⇑f) (f w) (f v)
    -/
    use w, v, h.symm, rfl
    /-
      🎉 no goals
    -/
  loopless a := by -- Porting note: `obviously` used to handle this
    /-
      V : Type u_1
      W : Type u_2
      X : Type u_3
      G✝ : SimpleGraph V
      G' : SimpleGraph W
      u v : V
      f : Function.Embedding V W
      G : SimpleGraph V
      a : W
      ⊢ Not (Relation.Map G.Adj (⇑f) (⇑f) a a)
    -/
    rintro ⟨v, w, h, rfl, h'⟩
    /-
      case intro.intro.intro.intro
      V : Type u_1
      W : Type u_2
      X : Type u_3
      G✝ : SimpleGraph V
      G' : SimpleGraph W
      u v✝ : V
      f : Function.Embedding V W
      G : SimpleGraph V
      v w : V
      h : G.Adj v w
      h' : Eq (f w) (f v)
      ⊢ False
    -/
    exact h.ne (f.injective h'.symm)
    /-
      🎉 no goals
    -/


instance instDecidableMapAdj {f : V ↪ W} {a b} [Decidable (Relation.Map G.Adj f f a b)] :
    Decidable ((G.map f).Adj a b) := ‹Decidable (Relation.Map G.Adj f f a b)›


@[simp]
theorem map_adj (f : V ↪ W) (G : SimpleGraph V) (u v : W) :
    (G.map f).Adj u v ↔ ∃ u' v' : V, G.Adj u' v' ∧ f u' = u ∧ f v' = v :=
  Iff.rfl


lemma map_adj_apply {G : SimpleGraph V} {f : V ↪ W} {a b : V} :
                                                /-
                                                  V : Type u_1
                                                  W : Type u_2
                                                  G : SimpleGraph V
                                                  f : Function.Embedding V W
                                                  a b : V
                                                  ⊢ Iff ((SimpleGraph.map f G).Adj (f a) (f b)) (G.Adj a b)
                                                -/
    (G.map f).Adj (f a) (f b) ↔ G.Adj a b := by simp
                                                /-
                                                  🎉 no goals
                                                -/


theorem map_monotone (f : V ↪ W) : Monotone (SimpleGraph.map f) := by
  /-
    V : Type u_1
    W : Type u_2
    f : Function.Embedding V W
    ⊢ Monotone (SimpleGraph.map f)
  -/
  rintro G G' h _ _ ⟨u, v, ha, rfl, rfl⟩
  /-
    case intro.intro.intro.intro
    V : Type u_1
    W : Type u_2
    f : Function.Embedding V W
    G G' : SimpleGraph V
    h : LE.le G G'
    u v : V
    ha : G.Adj u v
    ⊢ (SimpleGraph.map f G').Adj (f u) (f v)
  -/
  exact ⟨_, _, h ha, rfl, rfl⟩
  /-
    🎉 no goals
  -/


@[simp] lemma map_id : G.map (Function.Embedding.refl _) = G :=
  SimpleGraph.ext <| Relation.map_id_id _


@[simp] lemma map_map (f : V ↪ W) (g : W ↪ X) : (G.map f).map g = G.map (f.trans g) :=
  SimpleGraph.ext <| Relation.map_map _ _ _ _ _


/-- Given a function, there is a contravariant induced map on graphs by pulling back the
adjacency relation.
This is one of the ways of creating induced graphs. See `SimpleGraph.induce` for a wrapper.

This is surjective when `f` is injective (see `SimpleGraph.comap_surjective`). -/
protected def comap (f : V → W) (G : SimpleGraph W) : SimpleGraph V where
  Adj u v := G.Adj (f u) (f v)
  symm _ _ h := h.symm
  loopless _ := G.loopless _


@[simp] lemma comap_adj {G : SimpleGraph W} {f : V → W} :
    (G.comap f).Adj u v ↔ G.Adj (f u) (f v) := Iff.rfl


@[simp] lemma comap_id {G : SimpleGraph V} : G.comap id = G := SimpleGraph.ext rfl


@[simp] lemma comap_comap {G : SimpleGraph X} (f : V → W) (g : W → X) :
  (G.comap g).comap f = G.comap (g ∘ f) := rfl


instance instDecidableComapAdj (f : V → W) (G : SimpleGraph W) [DecidableRel G.Adj] :
    DecidableRel (G.comap f).Adj := fun _ _ ↦ ‹DecidableRel G.Adj› _ _


lemma comap_symm (G : SimpleGraph V) (e : V ≃ W) :
    G.comap e.symm.toEmbedding = G.map e.toEmbedding := by
  /-
    V : Type u_1
    W : Type u_2
    G : SimpleGraph V
    e : Equiv V W
    ⊢ Eq (SimpleGraph.comap (⇑e.symm.toEmbedding) G) (SimpleGraph.map e.toEmbeddin …
  -/
  ext; simp only [Equiv.apply_eq_iff_eq_symm_apply, comap_adj, map_adj, Equiv.toEmbedding_apply,
    exists_eq_right_right, exists_eq_right]


lemma map_symm (G : SimpleGraph W) (e : V ≃ W) :
                                                           /-
                                                             V : Type u_1
                                                             W : Type u_2
                                                             G : SimpleGraph W
                                                             e : Equiv V W
                                                             ⊢ Eq (SimpleGraph.map e.symm.toEmbedding G) (SimpleGraph.comap (⇑e.toEmbedding …
                                                           -/
    G.map e.symm.toEmbedding = G.comap e.toEmbedding := by rw [← comap_symm, e.symm_symm]
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem comap_monotone (f : V ↪ W) : Monotone (SimpleGraph.comap f) := by
  /-
    V : Type u_1
    W : Type u_2
    f : Function.Embedding V W
    ⊢ Monotone (SimpleGraph.comap ⇑f)
  -/
  intro G G' h _ _ ha
  /-
    V : Type u_1
    W : Type u_2
    f : Function.Embedding V W
    G G' : SimpleGraph W
    h : LE.le G G'
    v✝ w✝ : V
    ha : (SimpleGraph.comap (⇑f) G).Adj v✝ w✝
    ⊢ (SimpleGraph.comap (⇑f) G').Adj v✝ w✝
  -/
  exact h ha
  /-
    🎉 no goals
  -/


@[simp]
theorem comap_map_eq (f : V ↪ W) (G : SimpleGraph V) : (G.map f).comap f = G := by
  /-
    V : Type u_1
    W : Type u_2
    f : Function.Embedding V W
    G : SimpleGraph V
    ⊢ Eq (SimpleGraph.comap (⇑f) (SimpleGraph.map f G)) G
  -/
  ext
  /-
    case Adj.h.h.a
    V : Type u_1
    W : Type u_2
    f : Function.Embedding V W
    G : SimpleGraph V
    x✝¹ x✝ : V
    ⊢ Iff ((SimpleGraph.comap (⇑f) (SimpleGraph.map f G)).Adj x✝¹ x✝) (G.Adj x✝¹ x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem leftInverse_comap_map (f : V ↪ W) :
    Function.LeftInverse (SimpleGraph.comap f) (SimpleGraph.map f) :=
  comap_map_eq f


theorem map_injective (f : V ↪ W) : Function.Injective (SimpleGraph.map f) :=
  (leftInverse_comap_map f).injective


theorem comap_surjective (f : V ↪ W) : Function.Surjective (SimpleGraph.comap f) :=
  (leftInverse_comap_map f).surjective


theorem map_le_iff_le_comap (f : V ↪ W) (G : SimpleGraph V) (G' : SimpleGraph W) :
    G.map f ≤ G' ↔ G ≤ G'.comap f :=
  ⟨fun h _ _ ha => h ⟨_, _, ha, rfl, rfl⟩, by
    /-
      V : Type u_1
      W : Type u_2
      f : Function.Embedding V W
      G : SimpleGraph V
      G' : SimpleGraph W
      ⊢ LE.le G (SimpleGraph.comap (⇑f) G') → LE.le (SimpleGraph.map f G) G'
    -/
    rintro h _ _ ⟨u, v, ha, rfl, rfl⟩
    /-
      case intro.intro.intro.intro
      V : Type u_1
      W : Type u_2
      f : Function.Embedding V W
      G : SimpleGraph V
      G' : SimpleGraph W
      h : LE.le G (SimpleGraph.comap (⇑f) G')
      u v : V
      ha : G.Adj u v
      ⊢ G'.Adj (f u) (f v)
    -/
    exact h ha⟩
    /-
      🎉 no goals
    -/


theorem map_comap_le (f : V ↪ W) (G : SimpleGraph W) : (G.comap f).map f ≤ G := by
  /-
    V : Type u_1
    W : Type u_2
    f : Function.Embedding V W
    G : SimpleGraph W
    ⊢ LE.le (SimpleGraph.map f (SimpleGraph.comap (⇑f) G)) G
  -/
  rw [map_le_iff_le_comap]
  /-
    🎉 no goals
  -/


lemma le_comap_of_subsingleton (f : V → W) [Subsingleton V] : G ≤ G'.comap f := by
  /-
    V : Type u_1
    W : Type u_2
    G : SimpleGraph V
    G' : SimpleGraph W
    f : V → W
    inst✝ : Subsingleton V
    ⊢ LE.le G (SimpleGraph.comap f G')
  -/
  intros v w; simp [Subsingleton.elim v w]
              /-
                🎉 no goals
              -/


lemma map_le_of_subsingleton (f : V ↪ W) [Subsingleton V] : G.map f ≤ G' := by
  /-
    V : Type u_1
    W : Type u_2
    G : SimpleGraph V
    G' : SimpleGraph W
    f : Function.Embedding V W
    inst✝ : Subsingleton V
    ⊢ LE.le (SimpleGraph.map f G) G'
  -/
  rw [map_le_iff_le_comap]; apply le_comap_of_subsingleton
                            /-
                              🎉 no goals
                            -/


/-- Given a family of vertex types indexed by `ι`, pulling back from `⊤ : SimpleGraph ι`
yields the complete multipartite graph on the family.
Two vertices are adjacent if and only if their indices are not equal. -/
abbrev completeMultipartiteGraph {ι : Type*} (V : ι → Type*) : SimpleGraph (Σ i, V i) :=
  SimpleGraph.comap Sigma.fst ⊤


/-- Equivalent types have equivalent simple graphs. -/
@[simps apply]
protected def _root_.Equiv.simpleGraph (e : V ≃ W) : SimpleGraph V ≃ SimpleGraph W where
  toFun := SimpleGraph.comap e.symm
  invFun := SimpleGraph.comap e
                   /-
                     V : Type u_1
                     W : Type u_2
                     X : Type u_3
                     G : SimpleGraph V
                     G' : SimpleGraph W
                     u v : V
                     e : Equiv V W
                     x✝ : SimpleGraph V
                     ⊢ Eq (SimpleGraph.comap (⇑e) (SimpleGraph.comap (⇑e.symm) x✝)) x✝
                   -/
  left_inv _ := by simp
                   /-
                     🎉 no goals
                   -/
                    /-
                      V : Type u_1
                      W : Type u_2
                      X : Type u_3
                      G : SimpleGraph V
                      G' : SimpleGraph W
                      u v : V
                      e : Equiv V W
                      x✝ : SimpleGraph W
                      ⊢ Eq (SimpleGraph.comap (⇑e.symm) (SimpleGraph.comap (⇑e) x✝)) x✝
                    -/
  right_inv _ := by simp
                    /-
                      🎉 no goals
                    -/


@[simp] lemma _root_.Equiv.simpleGraph_refl : (Equiv.refl V).simpleGraph = Equiv.refl _ := by
  /-
    V : Type u_1
    ⊢ Eq (Equiv.refl V).simpleGraph (Equiv.refl (SimpleGraph V))
  -/
  ext; rfl
       /-
         🎉 no goals
       -/


@[simp] lemma _root_.Equiv.simpleGraph_trans (e₁ : V ≃ W) (e₂ : W ≃ X) :
  (e₁.trans e₂).simpleGraph = e₁.simpleGraph.trans e₂.simpleGraph := rfl


@[simp]
lemma _root_.Equiv.symm_simpleGraph (e : V ≃ W) : e.simpleGraph.symm = e.symm.simpleGraph := rfl


/-- Restrict a graph to the vertices in the set `s`, deleting all edges incident to vertices
outside the set. This is a wrapper around `SimpleGraph.comap`. -/
abbrev induce (s : Set V) (G : SimpleGraph V) : SimpleGraph s :=
  G.comap (Function.Embedding.subtype _)


@[simp] lemma induce_singleton_eq_top (v : V) : G.induce {v} = ⊤ := by
  /-
    V : Type u_1
    G : SimpleGraph V
    v : V
    ⊢ Eq (SimpleGraph.induce (Singleton.singleton v) G) Top.top
  -/
  rw [eq_top_iff]; apply le_comap_of_subsingleton
                   /-
                     🎉 no goals
                   -/


/-- Given a graph on a set of vertices, we can make it be a `SimpleGraph V` by
adding in the remaining vertices without adding in any additional edges.
This is a wrapper around `SimpleGraph.map`. -/
abbrev spanningCoe {s : Set V} (G : SimpleGraph s) : SimpleGraph V :=
  G.map (Function.Embedding.subtype _)


theorem induce_spanningCoe {s : Set V} {G : SimpleGraph s} : G.spanningCoe.induce s = G :=
  comap_map_eq _ _


theorem spanningCoe_induce_le (s : Set V) : (G.induce s).spanningCoe ≤ G :=
  map_comap_le _ _


/-- A graph homomorphism is a map on vertex sets that respects adjacency relations.

The notation `G →g G'` represents the type of graph homomorphisms. -/
abbrev Hom :=
  RelHom G.Adj G'.Adj


/-- A graph embedding is an embedding `f` such that for vertices `v w : V`,
`G'.Adj (f v) (f w) ↔ G.Adj v w`. Its image is an induced subgraph of G'.

The notation `G ↪g G'` represents the type of graph embeddings. -/
abbrev Embedding :=
  RelEmbedding G.Adj G'.Adj


/-- A graph isomorphism is a bijective map on vertex sets that respects adjacency relations.

The notation `G ≃g G'` represents the type of graph isomorphisms.
-/
abbrev Iso :=
  RelIso G.Adj G'.Adj


@[inherit_doc] infixl:50 " →g " => Hom

@[inherit_doc] infixl:50 " ↪g " => Embedding

@[inherit_doc] infixl:50 " ≃g " => Iso


/-- The identity homomorphism from a graph to itself. -/
protected abbrev id : G →g G :=
  RelHom.id _


@[simp, norm_cast] lemma coe_id : ⇑(Hom.id : G →g G) = id := rfl


instance [Subsingleton (V → W)] : Subsingleton (G →g H) := DFunLike.coe_injective.subsingleton


instance [IsEmpty V] : Unique (G →g H) where
  default := ⟨isEmptyElim, fun {a} ↦ isEmptyElim a⟩
  uniq _ := Subsingleton.elim _ _


instance [Finite V] [Finite W] : Finite (G →g H) := DFunLike.finite _


theorem map_adj {v w : V} (h : G.Adj v w) : G'.Adj (f v) (f w) :=
  f.map_rel' h


theorem map_mem_edgeSet {e : Sym2 V} (h : e ∈ G.edgeSet) : e.map f ∈ G'.edgeSet :=
  Sym2.ind (fun _ _ => f.map_rel') e h


theorem apply_mem_neighborSet {v w : V} (h : w ∈ G.neighborSet v) : f w ∈ G'.neighborSet (f v) :=
  map_adj f h


/-- The map between edge sets induced by a homomorphism.
The underlying map on edges is given by `Sym2.map`. -/
@[simps]
def mapEdgeSet (e : G.edgeSet) : G'.edgeSet :=
  ⟨Sym2.map f e, f.map_mem_edgeSet e.property⟩


/-- The map between neighbor sets induced by a homomorphism. -/
@[simps]
def mapNeighborSet (v : V) (w : G.neighborSet v) : G'.neighborSet (f v) :=
  ⟨f w, f.apply_mem_neighborSet w.property⟩


/-- The map between darts induced by a homomorphism. -/
def mapDart (d : G.Dart) : G'.Dart :=
  ⟨d.1.map f f, f.map_adj d.2⟩


@[simp]
theorem mapDart_apply (d : G.Dart) : f.mapDart d = ⟨d.1.map f f, f.map_adj d.2⟩ :=
  rfl


/-- The induced map for spanning subgraphs, which is the identity on vertices. -/
@[simps]
def mapSpanningSubgraphs {G G' : SimpleGraph V} (h : G ≤ G') : G →g G' where
  toFun x := x
  map_rel' ha := h ha


theorem mapEdgeSet.injective (hinj : Function.Injective f) : Function.Injective f.mapEdgeSet := by
  /-
    V : Type u_1
    W : Type u_2
    G : SimpleGraph V
    G' : SimpleGraph W
    f : G.Hom G'
    hinj : Function.Injective ⇑f
    ⊢ Function.Injective f.mapEdgeSet
  -/
  rintro ⟨e₁, h₁⟩ ⟨e₂, h₂⟩
  /-
    case mk.mk
    V : Type u_1
    W : Type u_2
    G : SimpleGraph V
    G' : SimpleGraph W
    f : G.Hom G'
    hinj : Function.Injective ⇑f
    e₁ : Sym2 V
    h₁ : Membership.mem G.edgeSet e₁
    e₂ : Sym2 V
    h₂ : Membership.mem G.edgeSet e₂
    ⊢ Eq (f.mapEdgeSet ⟨e₁, h₁⟩) (f.mapEdgeSet ⟨e₂, h₂⟩) → Eq ⟨e₁, h₁⟩ ⟨e₂, h₂⟩
  -/
  dsimp [Hom.mapEdgeSet]
  /-
    case mk.mk
    V : Type u_1
    W : Type u_2
    G : SimpleGraph V
    G' : SimpleGraph W
    f : G.Hom G'
    hinj : Function.Injective ⇑f
    e₁ : Sym2 V
    h₁ : Membership.mem G.edgeSet e₁
    e₂ : Sym2 V
    h₂ : Membership.mem G.edgeSet e₂
    ⊢ Eq ⟨Sym2.map (⇑f) e₁, ⋯⟩ ⟨Sym2.map (⇑f) e₂, ⋯⟩ → Eq ⟨e₁, h₁⟩ ⟨e₂, h₂⟩
  -/
  repeat rw [Subtype.mk_eq_mk]
  /-
    case mk.mk
    V : Type u_1
    W : Type u_2
    G : SimpleGraph V
    G' : SimpleGraph W
    f : G.Hom G'
    hinj : Function.Injective ⇑f
    e₁ : Sym2 V
    h₁ : Membership.mem G.edgeSet e₁
    e₂ : Sym2 V
    h₂ : Membership.mem G.edgeSet e₂
    ⊢ Eq (Sym2.map (⇑f) e₁) (Sym2.map (⇑f) e₂) → Eq e₁ e₂
  -/
  apply Sym2.map.injective hinj
  /-
    🎉 no goals
  -/


/-- Every graph homomorphism from a complete graph is injective. -/
theorem injective_of_top_hom (f : (⊤ : SimpleGraph V) →g G') : Function.Injective f := by
  /-
    V : Type u_1
    W : Type u_2
    G' : SimpleGraph W
    f : Top.top.Hom G'
    ⊢ Function.Injective ⇑f
  -/
  intro v w h
  /-
    V : Type u_1
    W : Type u_2
    G' : SimpleGraph W
    f : Top.top.Hom G'
    v w : V
    h : Eq (f v) (f w)
    ⊢ Eq v w
  -/
  contrapose! h
  /-
    V : Type u_1
    W : Type u_2
    G' : SimpleGraph W
    f : Top.top.Hom G'
    v w : V
    h : Ne v w
    ⊢ Ne (f v) (f w)
  -/
  exact G'.ne_of_adj (map_adj _ ((top_adj _ _).mpr h))
  /-
    🎉 no goals
  -/


/-- There is a homomorphism to a graph from a comapped graph.
When the function is injective, this is an embedding (see `SimpleGraph.Embedding.comap`). -/
@[simps]
protected def comap (f : V → W) (G : SimpleGraph W) : G.comap f →g G where
  toFun := f
                 /-
                   V : Type u_1
                   W : Type u_2
                   X : Type u_3
                   G✝ : SimpleGraph V
                   G' : SimpleGraph W
                   u v : V
                   G₁ G₂ : SimpleGraph V
                   H : SimpleGraph W
                   f✝ : G✝.Hom G'
                   f : V → W
                   G : SimpleGraph W
                   ⊢ ∀ {a b : V}, (SimpleGraph.comap f G).Adj a b → G.Adj (f a) (f b)
                 -/
  map_rel' := by simp
                 /-
                   🎉 no goals
                 -/


/-- Composition of graph homomorphisms. -/
abbrev comp (f' : G' →g G'') (f : G →g G') : G →g G'' :=
  RelHom.comp f' f


@[simp]
theorem coe_comp (f' : G' →g G'') (f : G →g G') : ⇑(f'.comp f) = f' ∘ f :=
  rfl


/-- The graph homomorphism from a smaller graph to a bigger one. -/
def ofLE (h : G₁ ≤ G₂) : G₁ →g G₂ := ⟨id, @h⟩


@[simp, norm_cast] lemma coe_ofLE (h : G₁ ≤ G₂) : ⇑(ofLE h) = id := rfl


/-- The identity embedding from a graph to itself. -/
abbrev refl : G ↪g G :=
  RelEmbedding.refl _


/-- An embedding of graphs gives rise to a homomorphism of graphs. -/
abbrev toHom : G →g G' :=
  f.toRelHom


@[simp] lemma coe_toHom (f : G ↪g H) : ⇑f.toHom = f := rfl


@[simp] theorem map_adj_iff {v w : V} : G'.Adj (f v) (f w) ↔ G.Adj v w :=
  f.map_rel_iff


theorem map_mem_edgeSet_iff {e : Sym2 V} : e.map f ∈ G'.edgeSet ↔ e ∈ G.edgeSet :=
  Sym2.ind (fun _ _ => f.map_adj_iff) e


theorem apply_mem_neighborSet_iff {v w : V} : f w ∈ G'.neighborSet (f v) ↔ w ∈ G.neighborSet v :=
  map_adj_iff f


/-- A graph embedding induces an embedding of edge sets. -/
@[simps]
def mapEdgeSet : G.edgeSet ↪ G'.edgeSet where
  toFun := Hom.mapEdgeSet f
  inj' := Hom.mapEdgeSet.injective f.toRelHom f.injective


/-- A graph embedding induces an embedding of neighbor sets. -/
@[simps]
def mapNeighborSet (v : V) : G.neighborSet v ↪ G'.neighborSet (f v) where
  toFun w := ⟨f w, f.apply_mem_neighborSet_iff.mpr w.2⟩
  inj' := by
    /-
      V : Type u_1
      W : Type u_2
      X : Type u_3
      G : SimpleGraph V
      G' : SimpleGraph W
      u v✝ : V
      H : SimpleGraph W
      f : G.Embedding G'
      v : V
      ⊢ Function.Injective fun w => ⟨f ↑w, ⋯⟩
    -/
    rintro ⟨w₁, h₁⟩ ⟨w₂, h₂⟩ h
    /-
      case mk.mk
      V : Type u_1
      W : Type u_2
      X : Type u_3
      G : SimpleGraph V
      G' : SimpleGraph W
      u v✝ : V
      H : SimpleGraph W
      f : G.Embedding G'
      v w₁ : V
      h₁ : Membership.mem (G.neighborSet v) w₁
      w₂ : V
      h₂ : Membership.mem (G.neighborSet v) w₂
      h : Eq ((fun w => ⟨f ↑w, ⋯⟩) ⟨w₁, h₁⟩) ((fun w => ⟨f ↑w, ⋯⟩) ⟨w₂, h₂⟩)
      ⊢ Eq ⟨w₁, h₁⟩ ⟨w₂, h₂⟩
    -/
    rw [Subtype.mk_eq_mk] at h ⊢
    /-
      case mk.mk
      V : Type u_1
      W : Type u_2
      X : Type u_3
      G : SimpleGraph V
      G' : SimpleGraph W
      u v✝ : V
      H : SimpleGraph W
      f : G.Embedding G'
      v w₁ : V
      h₁ : Membership.mem (G.neighborSet v) w₁
      w₂ : V
      h₂ : Membership.mem (G.neighborSet v) w₂
      h : Eq (f ↑⟨w₁, h₁⟩) (f ↑⟨w₂, h₂⟩)
      ⊢ Eq w₁ w₂
    -/
    exact f.inj' h
    /-
      🎉 no goals
    -/


/-- Given an injective function, there is an embedding from the comapped graph into the original
graph. -/
-- Porting note: @[simps] does not work here since `f` is not a constructor application.
-- `@[simps toEmbedding]` could work, but Floris suggested writing `comap_apply` for now.
protected def comap (f : V ↪ W) (G : SimpleGraph W) : G.comap f ↪g G :=
                              /-
                                V : Type u_1
                                W : Type u_2
                                X : Type u_3
                                G✝ : SimpleGraph V
                                G' : SimpleGraph W
                                u v : V
                                H : SimpleGraph W
                                f✝ : G✝.Embedding G'
                                f : Function.Embedding V W
                                G : SimpleGraph W
                                ⊢ ∀ {a b : V}, Iff (G.Adj (f a) (f b)) ((SimpleGraph.comap (⇑f) G).Adj a b)
                              -/
  { f with map_rel_iff' := by simp }
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem comap_apply (f : V ↪ W) (G : SimpleGraph W) (v : V) :
    SimpleGraph.Embedding.comap f G v = f v := rfl


/-- Given an injective function, there is an embedding from a graph into the mapped graph. -/
-- Porting note: @[simps] does not work here since `f` is not a constructor application.
-- `@[simps toEmbedding]` could work, but Floris suggested writing `map_apply` for now.
protected def map (f : V ↪ W) (G : SimpleGraph V) : G ↪g G.map f :=
                              /-
                                V : Type u_1
                                W : Type u_2
                                X : Type u_3
                                G✝ : SimpleGraph V
                                G' : SimpleGraph W
                                u v : V
                                H : SimpleGraph W
                                f✝ : G✝.Embedding G'
                                f : Function.Embedding V W
                                G : SimpleGraph V
                                ⊢ ∀ {a b : V}, Iff ((SimpleGraph.map f G).Adj (f a) (f b)) (G.Adj a b)
                              -/
  { f with map_rel_iff' := by simp }
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem map_apply (f : V ↪ W) (G : SimpleGraph V) (v : V) :
    SimpleGraph.Embedding.map f G v = f v := rfl


/-- Induced graphs embed in the original graph.

Note that if `G.induce s = ⊤` (i.e., if `s` is a clique) then this gives the embedding of a
complete graph. -/
protected abbrev induce (s : Set V) : G.induce s ↪g G :=
  SimpleGraph.Embedding.comap (Function.Embedding.subtype _) G


/-- Graphs on a set of vertices embed in their `spanningCoe`. -/
protected abbrev spanningCoe {s : Set V} (G : SimpleGraph s) : G ↪g G.spanningCoe :=
  SimpleGraph.Embedding.map (Function.Embedding.subtype _) G


/-- Embeddings of types induce embeddings of complete graphs on those types. -/
protected def completeGraph {α β : Type*} (f : α ↪ β) :
    (⊤ : SimpleGraph α) ↪g (⊤ : SimpleGraph β) :=
                              /-
                                V : Type u_1
                                W : Type u_2
                                X : Type u_3
                                G : SimpleGraph V
                                G' : SimpleGraph W
                                u v : V
                                H : SimpleGraph W
                                f✝ : G.Embedding G'
                                α : Type u_4
                                β : Type u_5
                                f : Function.Embedding α β
                                ⊢ ∀ {a b : α}, Iff (Top.top.Adj (f a) (f b)) (Top.top.Adj a b)
                              -/
  { f with map_rel_iff' := by simp }
                              /-
                                🎉 no goals
                              -/


@[simp] lemma coe_completeGraph {α β : Type*} (f : α ↪ β) : ⇑(Embedding.completeGraph f) = f := rfl


/-- Composition of graph embeddings. -/
abbrev comp (f' : G' ↪g G'') (f : G ↪g G') : G ↪g G'' :=
  f.trans f'


@[simp]
theorem coe_comp (f' : G' ↪g G'') (f : G ↪g G') : ⇑(f'.comp f) = f' ∘ f :=
  rfl


/-- The restriction of a morphism of graphs to induced subgraphs. -/
def induceHom : G.induce s →g G'.induce t where
  toFun := Set.MapsTo.restrict φ s t φst
  map_rel' := φ.map_rel'


@[simp, norm_cast] lemma coe_induceHom : ⇑(induceHom φ φst) = Set.MapsTo.restrict φ s t φst :=
  rfl


@[simp] lemma induceHom_id (G : SimpleGraph V) (s) :
    induceHom (Hom.id : G →g G) (Set.mapsTo_id s) = Hom.id := by
  /-
    V : Type u_1
    G : SimpleGraph V
    s : Set V
    ⊢ Eq (SimpleGraph.induceHom SimpleGraph.Hom.id ⋯) SimpleGraph.Hom.id
  -/
  ext x
  /-
    case h.a
    V : Type u_1
    G : SimpleGraph V
    s : Set V
    x : ↑s
    ⊢ Eq ↑((SimpleGraph.induceHom SimpleGraph.Hom.id ⋯) x) ↑(SimpleGraph.Hom.id x)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp] lemma induceHom_comp :
    (induceHom ψ ψtr).comp (induceHom φ φst) = induceHom (ψ.comp φ) (ψtr.comp φst) := by
  /-
    V : Type u_1
    W : Type u_2
    X : Type u_3
    G : SimpleGraph V
    G' : SimpleGraph W
    G'' : SimpleGraph X
    s : Set V
    t : Set W
    r : Set X
    φ : G.Hom G'
    φst : Set.MapsTo (⇑φ) s t
    ψ : G'.Hom G''
    ψtr : Set.MapsTo (⇑ψ) t r
    ⊢ Eq ((SimpleGraph.induceHom ψ ψtr).comp (SimpleGraph.induceHom φ φst)) (Simpl …
  -/
  ext x
  /-
    case h.a
    V : Type u_1
    W : Type u_2
    X : Type u_3
    G : SimpleGraph V
    G' : SimpleGraph W
    G'' : SimpleGraph X
    s : Set V
    t : Set W
    r : Set X
    φ : G.Hom G'
    φst : Set.MapsTo (⇑φ) s t
    ψ : G'.Hom G''
    ψtr : Set.MapsTo (⇑ψ) t r
    x : ↑s
    ⊢ Eq ↑(((SimpleGraph.induceHom ψ ψtr).comp (SimpleGraph.induceHom φ φst)) x) ↑ …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma induceHom_injective (hi : Set.InjOn φ s) :
    Function.Injective (induceHom φ φst) := by
  /-
    V : Type u_1
    W : Type u_2
    G : SimpleGraph V
    G' : SimpleGraph W
    s : Set V
    t : Set W
    φ : G.Hom G'
    φst : Set.MapsTo (⇑φ) s t
    hi : Set.InjOn (⇑φ) s
    ⊢ Function.Injective ⇑(SimpleGraph.induceHom φ φst)
  -/
                                    /-
                                      🎉 no goals
                                    -/
  erw [Set.MapsTo.restrict_inj] <;> assumption
                                    /-
                                      🎉 no goals
                                    -/


/-- Given an inclusion of vertex subsets, the induced embedding on induced graphs.
This is not an abbreviation for `induceHom` since we get an embedding in this case. -/
def induceHomOfLE (h : s ≤ s') : G.induce s ↪g G.induce s' where
  toEmbedding := Set.embeddingOfSubset s s' h
                     /-
                       V : Type u_1
                       W : Type u_2
                       X : Type u_3
                       G : SimpleGraph V
                       G' : SimpleGraph W
                       u v : V
                       s s' : Set V
                       h✝ h : LE.le s s'
                       ⊢ ∀ {a b : ↑s}, Iff ((SimpleGraph.induce s' G).Adj ((s.embeddingOfSubset s' h) …
                     -/
  map_rel_iff' := by simp
                     /-
                       🎉 no goals
                     -/


@[simp] lemma induceHomOfLE_apply (v : s) : (G.induceHomOfLE h) v = Set.inclusion h v := rfl


@[simp] lemma induceHomOfLE_toHom :
    (G.induceHomOfLE h).toHom = induceHom (.id : G →g G) ((Set.mapsTo_id s).mono_right h) := by
  /-
    V : Type u_1
    G : SimpleGraph V
    s s' : Set V
    h : LE.le s s'
    ⊢ Eq (G.induceHomOfLE h).toHom (SimpleGraph.induceHom SimpleGraph.Hom.id ⋯)
  -/
  ext; simp
       /-
         🎉 no goals
       -/


/-- The identity isomorphism of a graph with itself. -/
abbrev refl : G ≃g G :=
  RelIso.refl _


/-- An isomorphism of graphs gives rise to an embedding of graphs. -/
abbrev toEmbedding : G ↪g G' :=
  f.toRelEmbedding


/-- An isomorphism of graphs gives rise to a homomorphism of graphs. -/
abbrev toHom : G →g G' :=
  f.toEmbedding.toHom


/-- The inverse of a graph isomorphism. -/
abbrev symm : G' ≃g G :=
  RelIso.symm f


theorem map_adj_iff {v w : V} : G'.Adj (f v) (f w) ↔ G.Adj v w :=
  f.map_rel_iff


@[simp]
theorem symm_toHom_comp_toHom : f.symm.toHom.comp f.toHom = Hom.id := by
  /-
    V : Type u_1
    W : Type u_2
    G : SimpleGraph V
    G' : SimpleGraph W
    f : G.Iso G'
    ⊢ Eq (f.symm.toHom.comp f.toHom) SimpleGraph.Hom.id
  -/
  ext v
  simp only [RelHom.comp_apply, RelEmbedding.coe_toRelHom, RelIso.coe_toRelEmbedding,
    RelIso.symm_apply_apply, RelHom.id_apply]


@[simp]
theorem toHom_comp_symm_toHom : f.toHom.comp f.symm.toHom = Hom.id := by
  /-
    V : Type u_1
    W : Type u_2
    G : SimpleGraph V
    G' : SimpleGraph W
    f : G.Iso G'
    ⊢ Eq (f.toHom.comp f.symm.toHom) SimpleGraph.Hom.id
  -/
  ext v
  simp only [RelHom.comp_apply, RelEmbedding.coe_toRelHom, RelIso.coe_toRelEmbedding,
    RelIso.apply_symm_apply, RelHom.id_apply]


/-- An isomorphism of graphs induces an equivalence of edge sets. -/
@[simps]
def mapEdgeSet : G.edgeSet ≃ G'.edgeSet where
  toFun := Hom.mapEdgeSet f
  invFun := Hom.mapEdgeSet f.symm
  left_inv := by
    /-
      V : Type u_1
      W : Type u_2
      X : Type u_3
      G : SimpleGraph V
      G' : SimpleGraph W
      u v : V
      f : G.Iso G'
      ⊢ Function.LeftInverse (SimpleGraph.Hom.mapEdgeSet (RelIso.toRelEmbedding f.sy …
    -/
    rintro ⟨e, h⟩
    simp only [Hom.mapEdgeSet, RelEmbedding.toRelHom, Embedding.toFun_eq_coe,
      RelEmbedding.coe_toEmbedding, RelIso.coe_toRelEmbedding, Sym2.map_map, comp_apply,
      Subtype.mk.injEq]
    /-
      case mk
      V : Type u_1
      W : Type u_2
      X : Type u_3
      G : SimpleGraph V
      G' : SimpleGraph W
      u v : V
      f : G.Iso G'
      e : Sym2 V
      h : Membership.mem G.edgeSet e
      ⊢ Eq (Sym2.map (fun x => { toFun := ⇑f.symm, map_rel' := ⋯ } ({ toFun := ⇑f, m …
    -/
    convert congr_fun Sym2.map_id e
    /-
      case h.e'_2.h
      V : Type u_1
      W : Type u_2
      X : Type u_3
      G : SimpleGraph V
      G' : SimpleGraph W
      u v : V
      f : G.Iso G'
      e : Sym2 V
      h : Membership.mem G.edgeSet e
      x✝ : V
      a✝ : Membership.mem e x✝
      ⊢ Eq ({ toFun := ⇑f.symm, map_rel' := ⋯ } ({ toFun := ⇑f, map_rel' := ⋯ } x✝)) …
    -/
    exact RelIso.symm_apply_apply _ _
    /-
      🎉 no goals
    -/
  right_inv := by
    /-
      V : Type u_1
      W : Type u_2
      X : Type u_3
      G : SimpleGraph V
      G' : SimpleGraph W
      u v : V
      f : G.Iso G'
      ⊢ Function.RightInverse (SimpleGraph.Hom.mapEdgeSet (RelIso.toRelEmbedding f.s …
    -/
    rintro ⟨e, h⟩
    simp only [Hom.mapEdgeSet, RelEmbedding.toRelHom, Embedding.toFun_eq_coe,
      RelEmbedding.coe_toEmbedding, RelIso.coe_toRelEmbedding, Sym2.map_map, comp_apply,
      Subtype.mk.injEq]
    /-
      case mk
      V : Type u_1
      W : Type u_2
      X : Type u_3
      G : SimpleGraph V
      G' : SimpleGraph W
      u v : V
      f : G.Iso G'
      e : Sym2 W
      h : Membership.mem G'.edgeSet e
      ⊢ Eq (Sym2.map (fun x => { toFun := ⇑f, map_rel' := ⋯ } ({ toFun := ⇑f.symm, m …
    -/
    convert congr_fun Sym2.map_id e
    /-
      case h.e'_2.h
      V : Type u_1
      W : Type u_2
      X : Type u_3
      G : SimpleGraph V
      G' : SimpleGraph W
      u v : V
      f : G.Iso G'
      e : Sym2 W
      h : Membership.mem G'.edgeSet e
      x✝ : W
      a✝ : Membership.mem e x✝
      ⊢ Eq ({ toFun := ⇑f, map_rel' := ⋯ } ({ toFun := ⇑f.symm, map_rel' := ⋯ } x✝)) …
    -/
    exact RelIso.apply_symm_apply _ _
    /-
      🎉 no goals
    -/


/-- A graph isomorphism induces an equivalence of neighbor sets. -/
@[simps]
def mapNeighborSet (v : V) : G.neighborSet v ≃ G'.neighborSet (f v) where
  toFun w := ⟨f w, f.apply_mem_neighborSet_iff.mpr w.2⟩
  invFun w :=
    ⟨f.symm w, by
      /-
        V : Type u_1
        W : Type u_2
        X : Type u_3
        G : SimpleGraph V
        G' : SimpleGraph W
        u v✝ : V
        f : G.Iso G'
        v : V
        w : ↑(G'.neighborSet (f v))
        ⊢ Membership.mem (G.neighborSet v) (f.symm ↑w)
      -/
      simpa [RelIso.symm_apply_apply] using f.symm.apply_mem_neighborSet_iff.mpr w.2⟩
      /-
        🎉 no goals
      -/
                   /-
                     V : Type u_1
                     W : Type u_2
                     X : Type u_3
                     G : SimpleGraph V
                     G' : SimpleGraph W
                     u v✝ : V
                     f : G.Iso G'
                     v : V
                     w : ↑(G.neighborSet v)
                     ⊢ Eq ((fun w => ⟨f.symm ↑w, ⋯⟩) ((fun w => ⟨f ↑w, ⋯⟩) w)) w
                   -/
  left_inv w := by simp
                   /-
                     🎉 no goals
                   -/
                    /-
                      V : Type u_1
                      W : Type u_2
                      X : Type u_3
                      G : SimpleGraph V
                      G' : SimpleGraph W
                      u v✝ : V
                      f : G.Iso G'
                      v : V
                      w : ↑(G'.neighborSet (f v))
                      ⊢ Eq ((fun w => ⟨f ↑w, ⋯⟩) ((fun w => ⟨f.symm ↑w, ⋯⟩) w)) w
                    -/
  right_inv w := by simp
                    /-
                      🎉 no goals
                    -/


include f in
theorem card_eq [Fintype V] [Fintype W] : Fintype.card V = Fintype.card W := by
  /-
    V : Type u_1
    W : Type u_2
    G : SimpleGraph V
    G' : SimpleGraph W
    f : G.Iso G'
    inst✝¹ : Fintype V
    inst✝ : Fintype W
    ⊢ Eq (Fintype.card V) (Fintype.card W)
  -/
  rw [← Fintype.ofEquiv_card f.toEquiv]
  /-
    V : Type u_1
    W : Type u_2
    G : SimpleGraph V
    G' : SimpleGraph W
    f : G.Iso G'
    inst✝¹ : Fintype V
    inst✝ : Fintype W
    ⊢ Eq (Fintype.card W) (Fintype.card W)
  -/
  convert rfl
  /-
    🎉 no goals
  -/


/-- Given a bijection, there is an embedding from the comapped graph into the original
graph. -/
-- Porting note: `@[simps]` does not work here anymore since `f` is not a constructor application.
-- `@[simps toEmbedding]` could work, but Floris suggested writing `comap_apply` for now.
protected def comap (f : V ≃ W) (G : SimpleGraph W) : G.comap f.toEmbedding ≃g G :=
                              /-
                                V : Type u_1
                                W : Type u_2
                                X : Type u_3
                                G✝ : SimpleGraph V
                                G' : SimpleGraph W
                                u v : V
                                f✝ : G✝.Iso G'
                                f : Equiv V W
                                G : SimpleGraph W
                                ⊢ ∀ {a b : V}, Iff (G.Adj (f a) (f b)) ((SimpleGraph.comap (⇑f.toEmbedding) G) …
                              -/
  { f with map_rel_iff' := by simp }
                              /-
                                🎉 no goals
                              -/


@[simp]
lemma comap_apply (f : V ≃ W) (G : SimpleGraph W) (v : V) :
    SimpleGraph.Iso.comap f G v = f v := rfl


@[simp]
lemma comap_symm_apply (f : V ≃ W) (G : SimpleGraph W) (w : W) :
    (SimpleGraph.Iso.comap f G).symm w = f.symm w := rfl


/-- Given an injective function, there is an embedding from a graph into the mapped graph. -/
-- Porting note: `@[simps]` does not work here anymore since `f` is not a constructor application.
-- `@[simps toEmbedding]` could work, but Floris suggested writing `map_apply` for now.
protected def map (f : V ≃ W) (G : SimpleGraph V) : G ≃g G.map f.toEmbedding :=
                              /-
                                V : Type u_1
                                W : Type u_2
                                X : Type u_3
                                G✝ : SimpleGraph V
                                G' : SimpleGraph W
                                u v : V
                                f✝ : G✝.Iso G'
                                f : Equiv V W
                                G : SimpleGraph V
                                ⊢ ∀ {a b : V}, Iff ((SimpleGraph.map f.toEmbedding G).Adj (f a) (f b)) (G.Adj  …
                              -/
  { f with map_rel_iff' := by simp }
                              /-
                                🎉 no goals
                              -/


@[simp]
lemma map_apply (f : V ≃ W) (G : SimpleGraph V) (v : V) :
    SimpleGraph.Iso.map f G v = f v := rfl


@[simp]
lemma map_symm_apply (f : V ≃ W) (G : SimpleGraph V) (w : W) :
    (SimpleGraph.Iso.map f G).symm w = f.symm w := rfl


/-- Equivalences of types induce isomorphisms of complete graphs on those types. -/
protected def completeGraph {α β : Type*} (f : α ≃ β) :
    (⊤ : SimpleGraph α) ≃g (⊤ : SimpleGraph β) :=
                              /-
                                V : Type u_1
                                W : Type u_2
                                X : Type u_3
                                G : SimpleGraph V
                                G' : SimpleGraph W
                                u v : V
                                f✝ : G.Iso G'
                                α : Type u_4
                                β : Type u_5
                                f : Equiv α β
                                ⊢ ∀ {a b : α}, Iff (Top.top.Adj (f a) (f b)) (Top.top.Adj a b)
                              -/
  { f with map_rel_iff' := by simp }
                              /-
                                🎉 no goals
                              -/


theorem toEmbedding_completeGraph {α β : Type*} (f : α ≃ β) :
    (Iso.completeGraph f).toEmbedding = Embedding.completeGraph f.toEmbedding :=
  rfl


/-- Composition of graph isomorphisms. -/
abbrev comp (f' : G' ≃g G'') (f : G ≃g G') : G ≃g G'' :=
  f.trans f'


@[simp]
theorem coe_comp (f' : G' ≃g G'') (f : G ≃g G') : ⇑(f'.comp f) = f' ∘ f :=
  rfl


/-- The graph induced on `Set.univ` is isomorphic to the original graph. -/
@[simps!]
def induceUnivIso (G : SimpleGraph V) : G.induce Set.univ ≃g G where
  toEquiv := Equiv.Set.univ V
  map_rel_iff' := by simp only [Equiv.Set.univ, Equiv.coe_fn_mk, comap_adj, Embedding.coe_subtype,
                                Subtype.forall, Set.mem_univ, forall_true_left, implies_true]


/-- Given a graph over a finite vertex type `V` and a proof `hc` that `Fintype.card V = n`,
`G.overFin n` is an isomorphic (as shown in `overFinIso`) graph over `Fin n`. -/
def overFin (hc : Fintype.card V = n) : SimpleGraph (Fin n) where
  Adj x y := G.Adj ((Fintype.equivFinOfCardEq hc).symm x) ((Fintype.equivFinOfCardEq hc).symm y)
                 /-
                   V : Type u_1
                   W : Type u_2
                   X : Type u_3
                   G : SimpleGraph V
                   G' : SimpleGraph W
                   u v : V
                   inst✝ : Fintype V
                   n : Nat
                   hc : Eq (Fintype.card V) n
                   x y : Fin n
                   ⊢ (fun x y => G.Adj ((Fintype.equivFinOfCardEq hc).symm x) ((Fintype.equivFinO …
                 -/
  symm x y := by simp_rw [adj_comm, imp_self]
                 /-
                   🎉 no goals
                 -/


/-- The isomorphism between `G` and `G.overFin hc`. -/
noncomputable def overFinIso (hc : Fintype.card V = n) : G ≃g G.overFin hc := by
  /-
    V : Type u_1
    W : Type u_2
    X : Type u_3
    G : SimpleGraph V
    G' : SimpleGraph W
    u v : V
    inst✝ : Fintype V
    n : Nat
    hc : Eq (Fintype.card V) n
    ⊢ G.Iso (G.overFin hc)
  -/
  use Fintype.equivFinOfCardEq hc; simp [overFin]
                                   /-
                                     🎉 no goals
                                   -/


