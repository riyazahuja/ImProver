/--
The subgraph `M` of `G` is a matching if every vertex of `M` is incident to exactly one edge in `M`.
We say that the vertices in `M.support` are *matched* or *saturated*.
-/
def IsMatching (M : Subgraph G) : Prop := ∀ ⦃v⦄, v ∈ M.verts → ∃! w, M.Adj v w


/-- Given a vertex, returns the unique edge of the matching it is incident to. -/
noncomputable def IsMatching.toEdge (h : M.IsMatching) (v : M.verts) : M.edgeSet :=
  ⟨s(v, (h v.property).choose), (h v.property).choose_spec.1⟩


theorem IsMatching.toEdge_eq_of_adj (h : M.IsMatching) (hv : v ∈ M.verts) (hvw : M.Adj v w) :
    h.toEdge ⟨v, hv⟩ = ⟨s(v, w), hvw⟩ := by
  /-
    V : Type u_1
    G : SimpleGraph V
    M : G.Subgraph
    v w : V
    h : M.IsMatching
    hv : Membership.mem M.verts v
    hvw : M.Adj v w
    ⊢ Eq (h.toEdge ⟨v, hv⟩) ⟨Sym2.mk { fst := v, snd := w }, hvw⟩
  -/
  simp only [IsMatching.toEdge, Subtype.mk_eq_mk]
  /-
    V : Type u_1
    G : SimpleGraph V
    M : G.Subgraph
    v w : V
    h : M.IsMatching
    hv : Membership.mem M.verts v
    hvw : M.Adj v w
    ⊢ Eq (Sym2.mk { fst := v, snd := Exists.choose ⋯ }) (Sym2.mk { fst := v, snd : …
  -/
  congr
  /-
    case e_p.e_snd
    V : Type u_1
    G : SimpleGraph V
    M : G.Subgraph
    v w : V
    h : M.IsMatching
    hv : Membership.mem M.verts v
    hvw : M.Adj v w
    ⊢ Eq (Exists.choose ⋯) w
  -/
  exact ((h (M.edge_vert hvw)).choose_spec.2 w hvw).symm
  /-
    🎉 no goals
  -/


theorem IsMatching.toEdge.surjective (h : M.IsMatching) : Surjective h.toEdge := by
  /-
    V : Type u_1
    G : SimpleGraph V
    M : G.Subgraph
    h : M.IsMatching
    ⊢ Function.Surjective h.toEdge
  -/
  rintro ⟨e, he⟩
  /-
    case mk
    V : Type u_1
    G : SimpleGraph V
    M : G.Subgraph
    h : M.IsMatching
    e : Sym2 V
    he : Membership.mem M.edgeSet e
    ⊢ Exists fun a => Eq (h.toEdge a) ⟨e, he⟩
  -/
  induction' e with x y
  /-
    case mk.h
    V : Type u_1
    G : SimpleGraph V
    M : G.Subgraph
    h : M.IsMatching
    x y : V
    he : Membership.mem M.edgeSet (Sym2.mk { fst := x, snd := y })
    ⊢ Exists fun a => Eq (h.toEdge a) ⟨Sym2.mk { fst := x, snd := y }, he⟩
  -/
  exact ⟨⟨x, M.edge_vert he⟩, h.toEdge_eq_of_adj _ he⟩
  /-
    🎉 no goals
  -/


theorem IsMatching.toEdge_eq_toEdge_of_adj (h : M.IsMatching)
    (hv : v ∈ M.verts) (hw : w ∈ M.verts) (ha : M.Adj v w) :
    h.toEdge ⟨v, hv⟩ = h.toEdge ⟨w, hw⟩ := by
  /-
    V : Type u_1
    G : SimpleGraph V
    M : G.Subgraph
    v w : V
    h : M.IsMatching
    hv : Membership.mem M.verts v
    hw : Membership.mem M.verts w
    ha : M.Adj v w
    ⊢ Eq (h.toEdge ⟨v, hv⟩) (h.toEdge ⟨w, hw⟩)
  -/
  rw [h.toEdge_eq_of_adj hv ha, h.toEdge_eq_of_adj hw (M.symm ha), Subtype.mk_eq_mk, Sym2.eq_swap]
  /-
    🎉 no goals
  -/


lemma IsMatching.map_ofLE (h : M.IsMatching) (hGG' : G ≤ G') :
    (M.map (Hom.ofLE hGG')).IsMatching := by
  /-
    V : Type u_1
    G G' : SimpleGraph V
    M : G.Subgraph
    h : M.IsMatching
    hGG' : LE.le G G'
    ⊢ (SimpleGraph.Subgraph.map (SimpleGraph.Hom.ofLE hGG') M).IsMatching
  -/
  intro _ hv
  /-
    V : Type u_1
    G G' : SimpleGraph V
    M : G.Subgraph
    h : M.IsMatching
    hGG' : LE.le G G'
    v✝ : V
    hv : Membership.mem (SimpleGraph.Subgraph.map (SimpleGraph.Hom.ofLE hGG') M).v …
    ⊢ ExistsUnique fun w => (SimpleGraph.Subgraph.map (SimpleGraph.Hom.ofLE hGG')  …
  -/
  obtain ⟨_, hv, hv'⟩ := Set.mem_image _ _ _ |>.mp hv
  /-
    case intro.intro
    V : Type u_1
    G G' : SimpleGraph V
    M : G.Subgraph
    h : M.IsMatching
    hGG' : LE.le G G'
    v✝ : V
    hv✝ : Membership.mem (SimpleGraph.Subgraph.map (SimpleGraph.Hom.ofLE hGG') M). …
    w✝ : V
    hv : Membership.mem M.verts w✝
    hv' : Eq ((SimpleGraph.Hom.ofLE hGG') w✝) v✝
    ⊢ ExistsUnique fun w => (SimpleGraph.Subgraph.map (SimpleGraph.Hom.ofLE hGG')  …
  -/
  obtain ⟨w, hw⟩ := h hv
  /-
    case intro.intro.intro
    V : Type u_1
    G G' : SimpleGraph V
    M : G.Subgraph
    h : M.IsMatching
    hGG' : LE.le G G'
    v✝ : V
    hv✝ : Membership.mem (SimpleGraph.Subgraph.map (SimpleGraph.Hom.ofLE hGG') M). …
    w✝ : V
    hv : Membership.mem M.verts w✝
    hv' : Eq ((SimpleGraph.Hom.ofLE hGG') w✝) v✝
    w : V
    hw : And ((fun w => M.Adj w✝ w) w) (∀ (y : V), (fun w => M.Adj w✝ w) y → Eq y w)
    ⊢ ExistsUnique fun w => (SimpleGraph.Subgraph.map (SimpleGraph.Hom.ofLE hGG')  …
  -/
  use w
  /-
    case h
    V : Type u_1
    G G' : SimpleGraph V
    M : G.Subgraph
    h : M.IsMatching
    hGG' : LE.le G G'
    v✝ : V
    hv✝ : Membership.mem (SimpleGraph.Subgraph.map (SimpleGraph.Hom.ofLE hGG') M). …
    w✝ : V
    hv : Membership.mem M.verts w✝
    hv' : Eq ((SimpleGraph.Hom.ofLE hGG') w✝) v✝
    w : V
    hw : And ((fun w => M.Adj w✝ w) w) (∀ (y : V), (fun w => M.Adj w✝ w) y → Eq y w)
    ⊢ And ((fun w => (SimpleGraph.Subgraph.map (SimpleGraph.Hom.ofLE hGG') M).Adj  …
  -/
  simpa using hv' ▸ hw
  /-
    🎉 no goals
  -/


lemma IsMatching.sup (hM : M.IsMatching) (hM' : M'.IsMatching)
    (hd : Disjoint M.support M'.support) : (M ⊔ M').IsMatching := by
  /-
    V : Type u_1
    G : SimpleGraph V
    M M' : G.Subgraph
    hM : M.IsMatching
    hM' : M'.IsMatching
    hd : Disjoint M.support M'.support
    ⊢ (Max.max M M').IsMatching
  -/
  intro v hv
  have aux {N N' : Subgraph G} (hN : N.IsMatching) (hd : Disjoint N.support N'.support)
    (hmN: v ∈ N.verts) : ∃! w, (N ⊔ N').Adj v w := by
    obtain ⟨w, hw⟩ := hN hmN
    use w
    refine ⟨sup_adj.mpr (.inl hw.1), ?_⟩
    intro y hy
    cases hy with
    | inl h => exact hw.2 y h
    | inr h =>
      rw [Set.disjoint_left] at hd
      simpa [(mem_support _).mpr ⟨w, hw.1⟩, (mem_support _).mpr ⟨y, h⟩] using @hd v
  cases Set.mem_or_mem_of_mem_union hv with
  | inl hmM => exact aux hM hd hmM
  | inr hmM' =>
    rw [sup_comm]
    exact aux hM' (Disjoint.symm hd) hmM'


lemma IsMatching.iSup {ι : Sort _} {f : ι → Subgraph G} (hM : (i : ι) → (f i).IsMatching)
    (hd : Pairwise fun i j ↦ Disjoint (f i).support (f j).support) :
    (⨆ i, f i).IsMatching := by
  /-
    V : Type u_1
    G : SimpleGraph V
    ι : Type u_3
    f : ι → G.Subgraph
    hM : ∀ (i : ι), (f i).IsMatching
    hd : Pairwise fun i j => Disjoint (f i).support (f j).support
    ⊢ (_root_.iSup fun i => f i).IsMatching
  -/
  intro v hv
  /-
    V : Type u_1
    G : SimpleGraph V
    ι : Type u_3
    f : ι → G.Subgraph
    hM : ∀ (i : ι), (f i).IsMatching
    hd : Pairwise fun i j => Disjoint (f i).support (f j).support
    v : V
    hv : Membership.mem (_root_.iSup fun i => f i).verts v
    ⊢ ExistsUnique fun w => (_root_.iSup fun i => f i).Adj v w
  -/
  obtain ⟨i , hi⟩ := Set.mem_iUnion.mp (verts_iSup ▸ hv)
  /-
    case intro
    V : Type u_1
    G : SimpleGraph V
    ι : Type u_3
    f : ι → G.Subgraph
    hM : ∀ (i : ι), (f i).IsMatching
    hd : Pairwise fun i j => Disjoint (f i).support (f j).support
    v : V
    hv : Membership.mem (_root_.iSup fun i => f i).verts v
    i : ι
    hi : Membership.mem (f i).verts v
    ⊢ ExistsUnique fun w => (_root_.iSup fun i => f i).Adj v w
  -/
  obtain ⟨w , hw⟩ := hM i hi
  /-
    case intro.intro
    V : Type u_1
    G : SimpleGraph V
    ι : Type u_3
    f : ι → G.Subgraph
    hM : ∀ (i : ι), (f i).IsMatching
    hd : Pairwise fun i j => Disjoint (f i).support (f j).support
    v : V
    hv : Membership.mem (_root_.iSup fun i => f i).verts v
    i : ι
    hi : Membership.mem (f i).verts v
    w : V
    hw : And ((fun w => (f i).Adj v w) w) (∀ (y : V), (fun w => (f i).Adj v w) y → …
    ⊢ ExistsUnique fun w => (_root_.iSup fun i => f i).Adj v w
  -/
  use w
  /-
    case h
    V : Type u_1
    G : SimpleGraph V
    ι : Type u_3
    f : ι → G.Subgraph
    hM : ∀ (i : ι), (f i).IsMatching
    hd : Pairwise fun i j => Disjoint (f i).support (f j).support
    v : V
    hv : Membership.mem (_root_.iSup fun i => f i).verts v
    i : ι
    hi : Membership.mem (f i).verts v
    w : V
    hw : And ((fun w => (f i).Adj v w) w) (∀ (y : V), (fun w => (f i).Adj v w) y → …
    ⊢ And ((fun w => (_root_.iSup fun i => f i).Adj v w) w) (∀ (y : V), (fun w =>  …
  -/
  refine ⟨iSup_adj.mpr ⟨i, hw.1⟩, ?_⟩
  /-
    case h
    V : Type u_1
    G : SimpleGraph V
    ι : Type u_3
    f : ι → G.Subgraph
    hM : ∀ (i : ι), (f i).IsMatching
    hd : Pairwise fun i j => Disjoint (f i).support (f j).support
    v : V
    hv : Membership.mem (_root_.iSup fun i => f i).verts v
    i : ι
    hi : Membership.mem (f i).verts v
    w : V
    hw : And ((fun w => (f i).Adj v w) w) (∀ (y : V), (fun w => (f i).Adj v w) y → …
    ⊢ ∀ (y : V), (fun w => (_root_.iSup fun i => f i).Adj v w) y → Eq y w
  -/
  intro y hy
  /-
    case h
    V : Type u_1
    G : SimpleGraph V
    ι : Type u_3
    f : ι → G.Subgraph
    hM : ∀ (i : ι), (f i).IsMatching
    hd : Pairwise fun i j => Disjoint (f i).support (f j).support
    v : V
    hv : Membership.mem (_root_.iSup fun i => f i).verts v
    i : ι
    hi : Membership.mem (f i).verts v
    w : V
    hw : And ((fun w => (f i).Adj v w) w) (∀ (y : V), (fun w => (f i).Adj v w) y → …
    y : V
    hy : (_root_.iSup fun i => f i).Adj v y
    ⊢ Eq y w
  -/
  obtain ⟨i' , hi'⟩ := iSup_adj.mp hy
  /-
    case h.intro
    V : Type u_1
    G : SimpleGraph V
    ι : Type u_3
    f : ι → G.Subgraph
    hM : ∀ (i : ι), (f i).IsMatching
    hd : Pairwise fun i j => Disjoint (f i).support (f j).support
    v : V
    hv : Membership.mem (_root_.iSup fun i => f i).verts v
    i : ι
    hi : Membership.mem (f i).verts v
    w : V
    hw : And ((fun w => (f i).Adj v w) w) (∀ (y : V), (fun w => (f i).Adj v w) y → …
    y : V
    hy : (_root_.iSup fun i => f i).Adj v y
    i' : ι
    hi' : (f i').Adj v y
    ⊢ Eq y w
  -/
  by_cases heq : i = i'
    /-
      case pos
      V : Type u_1
      G : SimpleGraph V
      ι : Type u_3
      f : ι → G.Subgraph
      hM : ∀ (i : ι), (f i).IsMatching
      hd : Pairwise fun i j => Disjoint (f i).support (f j).support
      v : V
      hv : Membership.mem (_root_.iSup fun i => f i).verts v
      i : ι
      hi : Membership.mem (f i).verts v
      w : V
      hw : And ((fun w => (f i).Adj v w) w) (∀ (y : V), (fun w => (f i).Adj v w) y → …
      y : V
      hy : (_root_.iSup fun i => f i).Adj v y
      i' : ι
      hi' : (f i').Adj v y
      heq : Eq i i'
      ⊢ Eq y w
    -/
  · exact hw.2 y (heq.symm ▸ hi')
    /-
      🎉 no goals
    -/
    /-
      case neg
      V : Type u_1
      G : SimpleGraph V
      ι : Type u_3
      f : ι → G.Subgraph
      hM : ∀ (i : ι), (f i).IsMatching
      hd : Pairwise fun i j => Disjoint (f i).support (f j).support
      v : V
      hv : Membership.mem (_root_.iSup fun i => f i).verts v
      i : ι
      hi : Membership.mem (f i).verts v
      w : V
      hw : And ((fun w => (f i).Adj v w) w) (∀ (y : V), (fun w => (f i).Adj v w) y → …
      y : V
      hy : (_root_.iSup fun i => f i).Adj v y
      i' : ι
      hi' : (f i').Adj v y
      heq : Not (Eq i i')
      ⊢ Eq y w
    -/
  · have := hd heq
    /-
      case neg
      V : Type u_1
      G : SimpleGraph V
      ι : Type u_3
      f : ι → G.Subgraph
      hM : ∀ (i : ι), (f i).IsMatching
      hd : Pairwise fun i j => Disjoint (f i).support (f j).support
      v : V
      hv : Membership.mem (_root_.iSup fun i => f i).verts v
      i : ι
      hi : Membership.mem (f i).verts v
      w : V
      hw : And ((fun w => (f i).Adj v w) w) (∀ (y : V), (fun w => (f i).Adj v w) y → …
      y : V
      hy : (_root_.iSup fun i => f i).Adj v y
      i' : ι
      hi' : (f i').Adj v y
      heq : Not (Eq i i')
      this : (fun i j => Disjoint (f i).support (f j).support) i i'
      ⊢ Eq y w
    -/
    simp only [Set.disjoint_left] at this
    /-
      case neg
      V : Type u_1
      G : SimpleGraph V
      ι : Type u_3
      f : ι → G.Subgraph
      hM : ∀ (i : ι), (f i).IsMatching
      hd : Pairwise fun i j => Disjoint (f i).support (f j).support
      v : V
      hv : Membership.mem (_root_.iSup fun i => f i).verts v
      i : ι
      hi : Membership.mem (f i).verts v
      w : V
      hw : And ((fun w => (f i).Adj v w) w) (∀ (y : V), (fun w => (f i).Adj v w) y → …
      y : V
      hy : (_root_.iSup fun i => f i).Adj v y
      i' : ι
      hi' : (f i').Adj v y
      heq : Not (Eq i i')
      this : ∀ ⦃a : V⦄, Membership.mem (f i).support a → Not (Membership.mem (f i'). …
      ⊢ Eq y w
    -/
    simpa [(mem_support _).mpr ⟨w, hw.1⟩, (mem_support _).mpr ⟨y, hi'⟩] using @this v
    /-
      🎉 no goals
    -/


lemma IsMatching.subgraphOfAdj (h : G.Adj v w) : (G.subgraphOfAdj h).IsMatching := by
  /-
    V : Type u_1
    G : SimpleGraph V
    v w : V
    h : G.Adj v w
    ⊢ (G.subgraphOfAdj h).IsMatching
  -/
  intro _ hv
  /-
    V : Type u_1
    G : SimpleGraph V
    v w : V
    h : G.Adj v w
    v✝ : V
    hv : Membership.mem (G.subgraphOfAdj h).verts v✝
    ⊢ ExistsUnique fun w_1 => (G.subgraphOfAdj h).Adj v✝ w_1
  -/
  rw [subgraphOfAdj_verts, Set.mem_insert_iff, Set.mem_singleton_iff] at hv
  cases hv with
  | inl => use w; aesop
  | inr => use v; aesop


lemma IsMatching.coeSubgraph {G' : Subgraph G} {M : Subgraph G'.coe} (hM : M.IsMatching) :
    M.coeSubgraph.IsMatching := by
  /-
    V : Type u_1
    G : SimpleGraph V
    G' : G.Subgraph
    M : G'.coe.Subgraph
    hM : M.IsMatching
    ⊢ (SimpleGraph.Subgraph.coeSubgraph M).IsMatching
  -/
  intro _ hv
  /-
    V : Type u_1
    G : SimpleGraph V
    G' : G.Subgraph
    M : G'.coe.Subgraph
    hM : M.IsMatching
    v✝ : V
    hv : Membership.mem (SimpleGraph.Subgraph.coeSubgraph M).verts v✝
    ⊢ ExistsUnique fun w => (SimpleGraph.Subgraph.coeSubgraph M).Adj v✝ w
  -/
  obtain ⟨w, hw⟩ := hM <| Set.mem_of_mem_image_val <| M.verts_coeSubgraph.symm ▸ hv
  /-
    case intro
    V : Type u_1
    G : SimpleGraph V
    G' : G.Subgraph
    M : G'.coe.Subgraph
    hM : M.IsMatching
    v✝ : V
    hv : Membership.mem (SimpleGraph.Subgraph.coeSubgraph M).verts v✝
    w : ↑G'.verts
    hw : And ((fun w => M.Adj ⟨v✝, ⋯⟩ w) w) (∀ (y : ↑G'.verts), (fun w => M.Adj ⟨v …
    ⊢ ExistsUnique fun w => (SimpleGraph.Subgraph.coeSubgraph M).Adj v✝ w
  -/
  use w
  /-
    case h
    V : Type u_1
    G : SimpleGraph V
    G' : G.Subgraph
    M : G'.coe.Subgraph
    hM : M.IsMatching
    v✝ : V
    hv : Membership.mem (SimpleGraph.Subgraph.coeSubgraph M).verts v✝
    w : ↑G'.verts
    hw : And ((fun w => M.Adj ⟨v✝, ⋯⟩ w) w) (∀ (y : ↑G'.verts), (fun w => M.Adj ⟨v …
    ⊢ And ((fun w => (SimpleGraph.Subgraph.coeSubgraph M).Adj v✝ w) ↑w) (∀ (y : V) …
  -/
  refine ⟨?_, fun y hy => ?_⟩
    /-
      case h.refine_1
      V : Type u_1
      G : SimpleGraph V
      G' : G.Subgraph
      M : G'.coe.Subgraph
      hM : M.IsMatching
      v✝ : V
      hv : Membership.mem (SimpleGraph.Subgraph.coeSubgraph M).verts v✝
      w : ↑G'.verts
      hw : And ((fun w => M.Adj ⟨v✝, ⋯⟩ w) w) (∀ (y : ↑G'.verts), (fun w => M.Adj ⟨v …
      ⊢ (fun w => (SimpleGraph.Subgraph.coeSubgraph M).Adj v✝ w) ↑w
    -/
  · obtain ⟨v, hv⟩ := (Set.mem_image _ _ _).mp <| M.verts_coeSubgraph.symm ▸ hv
    /-
      case h.refine_1.intro
      V : Type u_1
      G : SimpleGraph V
      G' : G.Subgraph
      M : G'.coe.Subgraph
      hM : M.IsMatching
      v✝ : V
      hv✝ : Membership.mem (SimpleGraph.Subgraph.coeSubgraph M).verts v✝
      w : ↑G'.verts
      hw : And ((fun w => M.Adj ⟨v✝, ⋯⟩ w) w) (∀ (y : ↑G'.verts), (fun w => M.Adj ⟨v …
      v : Subtype fun x => Membership.mem G'.verts x
      hv : And (Membership.mem M.verts v) (Eq (↑v) v✝)
      ⊢ (SimpleGraph.Subgraph.coeSubgraph M).Adj v✝ ↑w
    -/
    simp only [coeSubgraph_adj, Subtype.coe_eta, Subtype.coe_prop, exists_const]
    /-
      case h.refine_1.intro
      V : Type u_1
      G : SimpleGraph V
      G' : G.Subgraph
      M : G'.coe.Subgraph
      hM : M.IsMatching
      v✝ : V
      hv✝ : Membership.mem (SimpleGraph.Subgraph.coeSubgraph M).verts v✝
      w : ↑G'.verts
      hw : And ((fun w => M.Adj ⟨v✝, ⋯⟩ w) w) (∀ (y : ↑G'.verts), (fun w => M.Adj ⟨v …
      v : Subtype fun x => Membership.mem G'.verts x
      hv : And (Membership.mem M.verts v) (Eq (↑v) v✝)
      ⊢ Exists fun h => M.Adj ⟨v✝, ⋯⟩ w
    -/
    exact ⟨hv.2 ▸ v.2, hw.1⟩
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      V : Type u_1
      G : SimpleGraph V
      G' : G.Subgraph
      M : G'.coe.Subgraph
      hM : M.IsMatching
      v✝ : V
      hv : Membership.mem (SimpleGraph.Subgraph.coeSubgraph M).verts v✝
      w : ↑G'.verts
      hw : And ((fun w => M.Adj ⟨v✝, ⋯⟩ w) w) (∀ (y : ↑G'.verts), (fun w => M.Adj ⟨v …
      y : V
      hy : (fun w => (SimpleGraph.Subgraph.coeSubgraph M).Adj v✝ w) y
      ⊢ Eq y ↑w
    -/
  · obtain ⟨_, hw', hvw⟩ := (coeSubgraph_adj _ _ _).mp hy
    /-
      case h.refine_2.intro.intro
      V : Type u_1
      G : SimpleGraph V
      G' : G.Subgraph
      M : G'.coe.Subgraph
      hM : M.IsMatching
      v✝ : V
      hv : Membership.mem (SimpleGraph.Subgraph.coeSubgraph M).verts v✝
      w : ↑G'.verts
      hw : And ((fun w => M.Adj ⟨v✝, ⋯⟩ w) w) (∀ (y : ↑G'.verts), (fun w => M.Adj ⟨v …
      y : V
      hy : (fun w => (SimpleGraph.Subgraph.coeSubgraph M).Adj v✝ w) y
      w✝ : Membership.mem G'.verts v✝
      hw' : Membership.mem G'.verts y
      hvw : M.Adj ⟨v✝, w✝⟩ ⟨y, hw'⟩
      ⊢ Eq y ↑w
    -/
    rw [← hw.2 ⟨y, hw'⟩ hvw]
    /-
      🎉 no goals
    -/


lemma IsMatching.exists_of_disjoint_sets_of_equiv {s t : Set V} (h : Disjoint s t)
    (f : s ≃ t) (hadj : ∀ v : s, G.Adj v (f v)) :
    ∃ M : Subgraph G, M.verts = s ∪ t ∧ M.IsMatching := by
  use {
    verts := s ∪ t
    Adj := fun v w ↦ (∃ h : v ∈ s, f ⟨v, h⟩ = w) ∨ (∃ h : w ∈ s, f ⟨w, h⟩ = v)
    adj_sub := by
      intro v w h
      obtain (⟨hv, rfl⟩ | ⟨hw, rfl⟩) := h
      · exact hadj ⟨v, _⟩
      · exact (hadj ⟨w, _⟩).symm
    edge_vert := by aesop }

  /-
    case h
    V : Type u_1
    G : SimpleGraph V
    s t : Set V
    h : Disjoint s t
    f : Equiv ↑s ↑t
    hadj : ∀ (v : ↑s), G.Adj ↑v ↑(f v)
    ⊢ And (Eq { verts := Union.union s t, Adj := fun v w => Or (Exists fun h => Eq …
  -/
  simp only [Subgraph.IsMatching, Set.mem_union, true_and]
  /-
    case h
    V : Type u_1
    G : SimpleGraph V
    s t : Set V
    h : Disjoint s t
    f : Equiv ↑s ↑t
    hadj : ∀ (v : ↑s), G.Adj ↑v ↑(f v)
    ⊢ ∀ ⦃v : V⦄, Or (Membership.mem s v) (Membership.mem t v) → ExistsUnique fun w …
  -/
  intro v hv
  /-
    case h
    V : Type u_1
    G : SimpleGraph V
    s t : Set V
    h : Disjoint s t
    f : Equiv ↑s ↑t
    hadj : ∀ (v : ↑s), G.Adj ↑v ↑(f v)
    v : V
    hv : Or (Membership.mem s v) (Membership.mem t v)
    ⊢ ExistsUnique fun w => Or (Exists fun h => Eq (↑(f ⟨v, h⟩)) w) (Exists fun h  …
  -/
  cases' hv with hl hr
    /-
      case h.inl
      V : Type u_1
      G : SimpleGraph V
      s t : Set V
      h : Disjoint s t
      f : Equiv ↑s ↑t
      hadj : ∀ (v : ↑s), G.Adj ↑v ↑(f v)
      v : V
      hl : Membership.mem s v
      ⊢ ExistsUnique fun w => Or (Exists fun h => Eq (↑(f ⟨v, h⟩)) w) (Exists fun h  …
    -/
  · use f ⟨v, hl⟩
    /-
      case h
      V : Type u_1
      G : SimpleGraph V
      s t : Set V
      h : Disjoint s t
      f : Equiv ↑s ↑t
      hadj : ∀ (v : ↑s), G.Adj ↑v ↑(f v)
      v : V
      hl : Membership.mem s v
      ⊢ And ((fun w => Or (Exists fun h => Eq (↑(f ⟨v, h⟩)) w) (Exists fun h => Eq ( …
    -/
    simp only [hl, exists_const, true_or, exists_true_left, true_and]
    /-
      case h
      V : Type u_1
      G : SimpleGraph V
      s t : Set V
      h : Disjoint s t
      f : Equiv ↑s ↑t
      hadj : ∀ (v : ↑s), G.Adj ↑v ↑(f v)
      v : V
      hl : Membership.mem s v
      ⊢ ∀ (y : V), Or (Eq (↑(f ⟨v, ⋯⟩)) y) (Exists fun h => Eq (↑(f ⟨y, h⟩)) v) → Eq …
    -/
    rintro y (rfl | ⟨hys, rfl⟩)
      /-
        case h.inl
        V : Type u_1
        G : SimpleGraph V
        s t : Set V
        h : Disjoint s t
        f : Equiv ↑s ↑t
        hadj : ∀ (v : ↑s), G.Adj ↑v ↑(f v)
        v : V
        hl : Membership.mem s v
        ⊢ Eq ↑(f ⟨v, ⋯⟩) ↑(f ⟨v, hl⟩)
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case h.inr.intro
        V : Type u_1
        G : SimpleGraph V
        s t : Set V
        h : Disjoint s t
        f : Equiv ↑s ↑t
        hadj : ∀ (v : ↑s), G.Adj ↑v ↑(f v)
        y : V
        hys : Membership.mem s y
        hl : Membership.mem s ↑(f ⟨y, hys⟩)
        ⊢ Eq y ↑(f ⟨↑(f ⟨y, hys⟩), hl⟩)
      -/
    · exact (h.ne_of_mem hl (f ⟨y, hys⟩).coe_prop rfl).elim
      /-
        🎉 no goals
      -/
    /-
      case h.inr
      V : Type u_1
      G : SimpleGraph V
      s t : Set V
      h : Disjoint s t
      f : Equiv ↑s ↑t
      hadj : ∀ (v : ↑s), G.Adj ↑v ↑(f v)
      v : V
      hr : Membership.mem t v
      ⊢ ExistsUnique fun w => Or (Exists fun h => Eq (↑(f ⟨v, h⟩)) w) (Exists fun h  …
    -/
  · use f.symm ⟨v, hr⟩
    simp only [Subtype.coe_eta, Equiv.apply_symm_apply, Subtype.coe_prop, exists_const, or_true,
      true_and]
    /-
      case h
      V : Type u_1
      G : SimpleGraph V
      s t : Set V
      h : Disjoint s t
      f : Equiv ↑s ↑t
      hadj : ∀ (v : ↑s), G.Adj ↑v ↑(f v)
      v : V
      hr : Membership.mem t v
      ⊢ ∀ (y : V), Or (Exists fun h => Eq (↑(f ⟨v, h⟩)) y) (Exists fun h => Eq (↑(f  …
    -/
    rintro y (⟨hy, rfl⟩ | ⟨hy, rfl⟩)
      /-
        case h.inl.intro
        V : Type u_1
        G : SimpleGraph V
        s t : Set V
        h : Disjoint s t
        f : Equiv ↑s ↑t
        hadj : ∀ (v : ↑s), G.Adj ↑v ↑(f v)
        v : V
        hr : Membership.mem t v
        hy : Membership.mem s v
        ⊢ Eq ↑(f ⟨v, hy⟩) ↑(f.symm ⟨v, hr⟩)
      -/
    · exact (h.ne_of_mem hy hr rfl).elim
      /-
        🎉 no goals
      -/
      /-
        case h.inr.intro
        V : Type u_1
        G : SimpleGraph V
        s t : Set V
        h : Disjoint s t
        f : Equiv ↑s ↑t
        hadj : ∀ (v : ↑s), G.Adj ↑v ↑(f v)
        y : V
        hy : Membership.mem s y
        hr : Membership.mem t ↑(f ⟨y, hy⟩)
        ⊢ Eq y ↑(f.symm ⟨↑(f ⟨y, hy⟩), hr⟩)
      -/
    · simp
      /-
        🎉 no goals
      -/


protected lemma IsMatching.map {G' : SimpleGraph W} {M : Subgraph G} (f : G →g G')
    (hf : Injective f) (hM : M.IsMatching) : (M.map f).IsMatching := by
  /-
    V : Type u_1
    W : Type u_2
    G : SimpleGraph V
    G' : SimpleGraph W
    M : G.Subgraph
    f : G.Hom G'
    hf : Function.Injective ⇑f
    hM : M.IsMatching
    ⊢ (SimpleGraph.Subgraph.map f M).IsMatching
  -/
  rintro _ ⟨v, hv, rfl⟩
  /-
    case intro.intro
    V : Type u_1
    W : Type u_2
    G : SimpleGraph V
    G' : SimpleGraph W
    M : G.Subgraph
    f : G.Hom G'
    hf : Function.Injective ⇑f
    hM : M.IsMatching
    v : V
    hv : Membership.mem M.verts v
    ⊢ ExistsUnique fun w => (SimpleGraph.Subgraph.map f M).Adj (f v) w
  -/
  obtain ⟨v', hv'⟩ := hM hv
  /-
    case intro.intro.intro
    V : Type u_1
    W : Type u_2
    G : SimpleGraph V
    G' : SimpleGraph W
    M : G.Subgraph
    f : G.Hom G'
    hf : Function.Injective ⇑f
    hM : M.IsMatching
    v : V
    hv : Membership.mem M.verts v
    v' : V
    hv' : And ((fun w => M.Adj v w) v') (∀ (y : V), (fun w => M.Adj v w) y → Eq y  …
    ⊢ ExistsUnique fun w => (SimpleGraph.Subgraph.map f M).Adj (f v) w
  -/
  use f v'
  /-
    case h
    V : Type u_1
    W : Type u_2
    G : SimpleGraph V
    G' : SimpleGraph W
    M : G.Subgraph
    f : G.Hom G'
    hf : Function.Injective ⇑f
    hM : M.IsMatching
    v : V
    hv : Membership.mem M.verts v
    v' : V
    hv' : And ((fun w => M.Adj v w) v') (∀ (y : V), (fun w => M.Adj v w) y → Eq y  …
    ⊢ And ((fun w => (SimpleGraph.Subgraph.map f M).Adj (f v) w) (f v')) (∀ (y : W …
  -/
  refine ⟨⟨v, v', hv'.1, rfl, rfl⟩, ?_⟩
  /-
    case h
    V : Type u_1
    W : Type u_2
    G : SimpleGraph V
    G' : SimpleGraph W
    M : G.Subgraph
    f : G.Hom G'
    hf : Function.Injective ⇑f
    hM : M.IsMatching
    v : V
    hv : Membership.mem M.verts v
    v' : V
    hv' : And ((fun w => M.Adj v w) v') (∀ (y : V), (fun w => M.Adj v w) y → Eq y  …
    ⊢ ∀ (y : W), (fun w => (SimpleGraph.Subgraph.map f M).Adj (f v) w) y → Eq y (f …
  -/
  rintro _ ⟨w, w', hw, hw', rfl⟩
  /-
    case h.intro.intro.intro.intro
    V : Type u_1
    W : Type u_2
    G : SimpleGraph V
    G' : SimpleGraph W
    M : G.Subgraph
    f : G.Hom G'
    hf : Function.Injective ⇑f
    hM : M.IsMatching
    v : V
    hv : Membership.mem M.verts v
    v' : V
    hv' : And ((fun w => M.Adj v w) v') (∀ (y : V), (fun w => M.Adj v w) y → Eq y  …
    w w' : V
    hw : M.Adj w w'
    hw' : Eq (f w) (f v)
    ⊢ Eq (f w') (f v')
  -/
  cases hf hw'.symm
  /-
    case h.intro.intro.intro.intro.refl
    V : Type u_1
    W : Type u_2
    G : SimpleGraph V
    G' : SimpleGraph W
    M : G.Subgraph
    f : G.Hom G'
    hf : Function.Injective ⇑f
    hM : M.IsMatching
    v : V
    hv : Membership.mem M.verts v
    v' : V
    hv' : And ((fun w => M.Adj v w) v') (∀ (y : V), (fun w => M.Adj v w) y → Eq y  …
    w' : V
    hw : M.Adj v w'
    hw' : Eq (f v) (f v)
    ⊢ Eq (f w') (f v')
  -/
  rw [hv'.2 w' hw]
  /-
    🎉 no goals
  -/


@[simp]
lemma Iso.isMatching_map {G' : SimpleGraph W} {M : Subgraph G} (f : G ≃g G') :
    (M.map f.toHom).IsMatching ↔ M.IsMatching where
              /-
                V : Type u_1
                W : Type u_2
                G : SimpleGraph V
                G' : SimpleGraph W
                M : G.Subgraph
                f : G.Iso G'
                h : (SimpleGraph.Subgraph.map f.toHom M).IsMatching
                ⊢ M.IsMatching
              -/
   mp h := by simpa [← map_comp] using h.map f.symm.toHom f.symm.injective
              /-
                🎉 no goals
              -/
   mpr := .map f.toHom f.injective


/--
The subgraph `M` of `G` is a perfect matching on `G` if it's a matching and every vertex `G` is
matched.
-/
def IsPerfectMatching (M : G.Subgraph) : Prop := M.IsMatching ∧ M.IsSpanning


theorem IsMatching.support_eq_verts (h : M.IsMatching) : M.support = M.verts := by
  /-
    V : Type u_1
    G : SimpleGraph V
    M : G.Subgraph
    h : M.IsMatching
    ⊢ Eq M.support M.verts
  -/
  refine M.support_subset_verts.antisymm fun v hv => ?_
  /-
    V : Type u_1
    G : SimpleGraph V
    M : G.Subgraph
    h : M.IsMatching
    v : V
    hv : Membership.mem M.verts v
    ⊢ Membership.mem M.support v
  -/
  obtain ⟨w, hvw, -⟩ := h hv
  /-
    case intro.intro
    V : Type u_1
    G : SimpleGraph V
    M : G.Subgraph
    h : M.IsMatching
    v : V
    hv : Membership.mem M.verts v
    w : V
    hvw : M.Adj v w
    ⊢ Membership.mem M.support v
  -/
  exact ⟨_, hvw⟩
  /-
    🎉 no goals
  -/


theorem isMatching_iff_forall_degree [∀ v, Fintype (M.neighborSet v)] :
    M.IsMatching ↔ ∀ v : V, v ∈ M.verts → M.degree v = 1 := by
  /-
    V : Type u_1
    G : SimpleGraph V
    M : G.Subgraph
    inst✝ : (v : V) → Fintype ↑(M.neighborSet v)
    ⊢ Iff M.IsMatching (∀ (v : V), Membership.mem M.verts v → Eq (M.degree v) 1)
  -/
  simp only [degree_eq_one_iff_unique_adj, IsMatching]
  /-
    🎉 no goals
  -/


theorem IsMatching.even_card [Fintype M.verts] (h : M.IsMatching) : Even M.verts.toFinset.card := by
  classical
  rw [isMatching_iff_forall_degree] at h
  use M.coe.edgeFinset.card
  rw [← two_mul, ← M.coe.sum_degrees_eq_twice_card_edges]
  -- Porting note: `SimpleGraph.Subgraph.coe_degree` does not trigger because it uses
  -- instance arguments instead of implicit arguments for the first `Fintype` argument.
  -- Using a `convert_to` to swap out the `Fintype` instance to the "right" one.
  convert_to _ = Finset.sum Finset.univ fun v => SimpleGraph.degree (Subgraph.coe M) v using 3
  simp [h, Finset.card_univ]


theorem isPerfectMatching_iff : M.IsPerfectMatching ↔ ∀ v, ∃! w, M.Adj v w := by
  /-
    V : Type u_1
    G : SimpleGraph V
    M : G.Subgraph
    ⊢ Iff M.IsPerfectMatching (∀ (v : V), ExistsUnique fun w => M.Adj v w)
  -/
  refine ⟨?_, fun hm => ⟨fun v _ => hm v, fun v => ?_⟩⟩
    /-
      case refine_1
      V : Type u_1
      G : SimpleGraph V
      M : G.Subgraph
      ⊢ M.IsPerfectMatching → ∀ (v : V), ExistsUnique fun w => M.Adj v w
    -/
  · rintro ⟨hm, hs⟩ v
    /-
      case refine_1.intro
      V : Type u_1
      G : SimpleGraph V
      M : G.Subgraph
      hm : M.IsMatching
      hs : M.IsSpanning
      v : V
      ⊢ ExistsUnique fun w => M.Adj v w
    -/
    exact hm (hs v)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      V : Type u_1
      G : SimpleGraph V
      M : G.Subgraph
      hm : ∀ (v : V), ExistsUnique fun w => M.Adj v w
      v : V
      ⊢ Membership.mem M.verts v
    -/
  · obtain ⟨w, hw, -⟩ := hm v
    /-
      case refine_2.intro.intro
      V : Type u_1
      G : SimpleGraph V
      M : G.Subgraph
      hm : ∀ (v : V), ExistsUnique fun w => M.Adj v w
      v w : V
      hw : M.Adj v w
      ⊢ Membership.mem M.verts v
    -/
    exact M.edge_vert hw
    /-
      🎉 no goals
    -/


theorem isPerfectMatching_iff_forall_degree [∀ v, Fintype (M.neighborSet v)] :
    M.IsPerfectMatching ↔ ∀ v, M.degree v = 1 := by
  /-
    V : Type u_1
    G : SimpleGraph V
    M : G.Subgraph
    inst✝ : (v : V) → Fintype ↑(M.neighborSet v)
    ⊢ Iff M.IsPerfectMatching (∀ (v : V), Eq (M.degree v) 1)
  -/
  simp [degree_eq_one_iff_unique_adj, isPerfectMatching_iff]
  /-
    🎉 no goals
  -/


theorem IsPerfectMatching.even_card [Fintype V] (h : M.IsPerfectMatching) :
    Even (Fintype.card V) := by
  classical
  simpa only [h.2.card_verts] using IsMatching.even_card h.1


lemma IsMatching.induce_connectedComponent (h : M.IsMatching) (c : ConnectedComponent G) :
    (M.induce (M.verts ∩ c.supp)).IsMatching := by
  /-
    V : Type u_1
    G : SimpleGraph V
    M : G.Subgraph
    h : M.IsMatching
    c : G.ConnectedComponent
    ⊢ (M.induce (Inter.inter M.verts c.supp)).IsMatching
  -/
  intro _ hv
  /-
    V : Type u_1
    G : SimpleGraph V
    M : G.Subgraph
    h : M.IsMatching
    c : G.ConnectedComponent
    v✝ : V
    hv : Membership.mem (M.induce (Inter.inter M.verts c.supp)).verts v✝
    ⊢ ExistsUnique fun w => (M.induce (Inter.inter M.verts c.supp)).Adj v✝ w
  -/
  obtain ⟨hv, rfl⟩ := hv
  /-
    case intro
    V : Type u_1
    G : SimpleGraph V
    M : G.Subgraph
    h : M.IsMatching
    v✝ : V
    hv : Membership.mem M.verts v✝
    ⊢ ExistsUnique fun w => (M.induce (Inter.inter M.verts (G.connectedComponentMk …
  -/
  obtain ⟨w, hvw, hw⟩ := h hv
  /-
    case intro.intro.intro
    V : Type u_1
    G : SimpleGraph V
    M : G.Subgraph
    h : M.IsMatching
    v✝ : V
    hv : Membership.mem M.verts v✝
    w : V
    hvw : M.Adj v✝ w
    hw : ∀ (y : V), (fun w => M.Adj v✝ w) y → Eq y w
    ⊢ ExistsUnique fun w => (M.induce (Inter.inter M.verts (G.connectedComponentMk …
  -/
  use w
  /-
    case h
    V : Type u_1
    G : SimpleGraph V
    M : G.Subgraph
    h : M.IsMatching
    v✝ : V
    hv : Membership.mem M.verts v✝
    w : V
    hvw : M.Adj v✝ w
    hw : ∀ (y : V), (fun w => M.Adj v✝ w) y → Eq y w
    ⊢ And ((fun w => (M.induce (Inter.inter M.verts (G.connectedComponentMk v✝).su …
  -/
  simpa [hv, hvw, M.edge_vert hvw.symm, (M.adj_sub hvw).symm.reachable] using fun _ _ _ ↦ hw _
  /-
    🎉 no goals
  -/


lemma IsPerfectMatching.induce_connectedComponent_isMatching (h : M.IsPerfectMatching)
    (c : ConnectedComponent G) : (M.induce c.supp).IsMatching := by
  /-
    V : Type u_1
    G : SimpleGraph V
    M : G.Subgraph
    h : M.IsPerfectMatching
    c : G.ConnectedComponent
    ⊢ (M.induce c.supp).IsMatching
  -/
  simpa [h.2.verts_eq_univ] using h.1.induce_connectedComponent c
  /-
    🎉 no goals
  -/


@[simp]
lemma IsPerfectMatching.toSubgraph_spanningCoe_iff (h : M.spanningCoe ≤ G') :
    (G'.toSubgraph M.spanningCoe h).IsPerfectMatching ↔ M.IsPerfectMatching := by
  /-
    V : Type u_1
    G G' : SimpleGraph V
    M : G.Subgraph
    h : LE.le M.spanningCoe G'
    ⊢ Iff (SimpleGraph.toSubgraph M.spanningCoe h).IsPerfectMatching M.IsPerfectMa …
  -/
  simp only [isPerfectMatching_iff, toSubgraph_adj, spanningCoe_adj]
  /-
    🎉 no goals
  -/


lemma even_card_of_isPerfectMatching [Fintype V] [DecidableEq V] [DecidableRel G.Adj]
    (c : ConnectedComponent G) (hM : M.IsPerfectMatching) :
    Even (Fintype.card c.supp) := by
  #adaptation_note
  /--
  After https://github.com/leanprover/lean4/pull/5020, some instances that use the chain of coercions
  `[SetLike X], X → Set α → Sort _` are
  blocked by the discrimination tree. This can be fixed by redeclaring the instance for `X`
  using the double coercion but the proper fix seems to avoid the double coercion.
  -/
  /-
    V : Type u_1
    G : SimpleGraph V
    M : G.Subgraph
    inst✝² : Fintype V
    inst✝¹ : DecidableEq V
    inst✝ : DecidableRel G.Adj
    c : G.ConnectedComponent
    hM : M.IsPerfectMatching
    ⊢ Even (Fintype.card ↑c.supp)
  -/
  letI : DecidablePred fun x ↦ x ∈ (M.induce c.supp).verts := fun a ↦ G.instDecidableMemSupp c a
  /-
    V : Type u_1
    G : SimpleGraph V
    M : G.Subgraph
    inst✝² : Fintype V
    inst✝¹ : DecidableEq V
    inst✝ : DecidableRel G.Adj
    c : G.ConnectedComponent
    hM : M.IsPerfectMatching
    this : DecidablePred fun x => Membership.mem (M.induce c.supp).verts x := fun  …
    ⊢ Even (Fintype.card ↑c.supp)
  -/
  simpa using (hM.induce_connectedComponent_isMatching c).even_card
  /-
    🎉 no goals
  -/


lemma odd_matches_node_outside [Finite V] {u : Set V}
    {c : ConnectedComponent (Subgraph.deleteVerts ⊤ u).coe}
    (hM : M.IsPerfectMatching) (codd : Odd (Nat.card c.supp)) :
    ∃ᵉ (w ∈ u) (v : ((⊤ : G.Subgraph).deleteVerts u).verts), M.Adj v w ∧ v ∈ c.supp := by
  /-
    V : Type u_1
    G : SimpleGraph V
    M : G.Subgraph
    inst✝ : Finite V
    u : Set V
    c : (Top.top.deleteVerts u).coe.ConnectedComponent
    hM : M.IsPerfectMatching
    codd : Odd (Nat.card ↑c.supp)
    ⊢ Exists fun w => And (Membership.mem u w) (Exists fun v => And (M.Adj (↑v) w) …
  -/
  by_contra! h
  have hMmatch : (M.induce c.supp).IsMatching := by
    intro v hv
    obtain ⟨w, hw⟩ := hM.1 (hM.2 v)
    obtain ⟨⟨v', hv'⟩, ⟨hv , rfl⟩⟩ := hv
    use w
    have hwnu : w ∉ u := fun hw' ↦ h w hw' ⟨v', hv'⟩ (hw.1) hv
    refine ⟨⟨⟨⟨v', hv'⟩, hv, rfl⟩, ?_, hw.1⟩, fun _ hy ↦ hw.2 _ hy.2.2⟩
    apply ConnectedComponent.mem_coe_supp_of_adj ⟨⟨v', hv'⟩, ⟨hv, rfl⟩⟩ ⟨by trivial, hwnu⟩
    simp only [Subgraph.induce_verts, Subgraph.verts_top, Set.mem_diff, Set.mem_univ, true_and,
      Subgraph.induce_adj, hwnu, not_false_eq_true, and_self, Subgraph.top_adj, M.adj_sub hw.1,
      and_true] at hv' ⊢
    trivial
  /-
    V : Type u_1
    G : SimpleGraph V
    M : G.Subgraph
    inst✝ : Finite V
    u : Set V
    c : (Top.top.deleteVerts u).coe.ConnectedComponent
    hM : M.IsPerfectMatching
    codd : Odd (Nat.card ↑c.supp)
    h : ∀ (w : V), Membership.mem u w → ∀ (v : ↑(Top.top.deleteVerts u).verts), M. …
    hMmatch : (M.induce (Set.image Subtype.val c.supp)).IsMatching
    ⊢ False
  -/
  apply Nat.not_even_iff_odd.2 codd
  /-
    V : Type u_1
    G : SimpleGraph V
    M : G.Subgraph
    inst✝ : Finite V
    u : Set V
    c : (Top.top.deleteVerts u).coe.ConnectedComponent
    hM : M.IsPerfectMatching
    codd : Odd (Nat.card ↑c.supp)
    h : ∀ (w : V), Membership.mem u w → ∀ (v : ↑(Top.top.deleteVerts u).verts), M. …
    hMmatch : (M.induce (Set.image Subtype.val c.supp)).IsMatching
    ⊢ Even (Nat.card ↑c.supp)
  -/
  haveI : Fintype ↑(Subgraph.induce M (Subtype.val '' supp c)).verts := Fintype.ofFinite _
  classical
  have hMeven := Subgraph.IsMatching.even_card hMmatch
  haveI : Fintype (c.supp) := Fintype.ofFinite _
  simp only [Subgraph.induce_verts, Subgraph.verts_top, Set.toFinset_image,
    Nat.card_eq_fintype_card, Set.toFinset_image,
    Finset.card_image_of_injective _ (Subtype.val_injective), Set.toFinset_card] at hMeven ⊢
  exact hMeven


/--
A graph is matching free if it has no perfect matching. It does not make much sense to
consider a graph being free of just matchings, because any non-trivial graph has those.
-/
def IsMatchingFree (G : SimpleGraph V) := ∀ M : Subgraph G, ¬ M.IsPerfectMatching


lemma IsMatchingFree.mono {G G' : SimpleGraph V} (h : G ≤ G') (hmf : G'.IsMatchingFree) :
    G.IsMatchingFree := by
  /-
    V : Type u_1
    G G' : SimpleGraph V
    h : LE.le G G'
    hmf : G'.IsMatchingFree
    ⊢ G.IsMatchingFree
  -/
  intro x
  /-
    V : Type u_1
    G G' : SimpleGraph V
    h : LE.le G G'
    hmf : G'.IsMatchingFree
    x : G.Subgraph
    ⊢ Not x.IsPerfectMatching
  -/
  by_contra! hc
  /-
    V : Type u_1
    G G' : SimpleGraph V
    h : LE.le G G'
    hmf : G'.IsMatchingFree
    x : G.Subgraph
    hc : x.IsPerfectMatching
    ⊢ False
  -/
  apply hmf (x.map (SimpleGraph.Hom.ofLE h))
  /-
    V : Type u_1
    G G' : SimpleGraph V
    h : LE.le G G'
    hmf : G'.IsMatchingFree
    x : G.Subgraph
    hc : x.IsPerfectMatching
    ⊢ (SimpleGraph.Subgraph.map (SimpleGraph.Hom.ofLE h) x).IsPerfectMatching
  -/
  refine ⟨hc.1.map_ofLE h, ?_⟩
  /-
    V : Type u_1
    G G' : SimpleGraph V
    h : LE.le G G'
    hmf : G'.IsMatchingFree
    x : G.Subgraph
    hc : x.IsPerfectMatching
    ⊢ (SimpleGraph.Subgraph.map (SimpleGraph.Hom.ofLE h) x).IsSpanning
  -/
  intro v
  /-
    V : Type u_1
    G G' : SimpleGraph V
    h : LE.le G G'
    hmf : G'.IsMatchingFree
    x : G.Subgraph
    hc : x.IsPerfectMatching
    v : V
    ⊢ Membership.mem (SimpleGraph.Subgraph.map (SimpleGraph.Hom.ofLE h) x).verts v
  -/
  simp only [Subgraph.map_verts, Hom.coe_ofLE, id_eq, Set.image_id']
  /-
    V : Type u_1
    G G' : SimpleGraph V
    h : LE.le G G'
    hmf : G'.IsMatchingFree
    x : G.Subgraph
    hc : x.IsPerfectMatching
    v : V
    ⊢ Membership.mem x.verts v
  -/
  exact hc.2 v
  /-
    🎉 no goals
  -/


lemma exists_maximal_isMatchingFree [Finite V] (h : G.IsMatchingFree) :
    ∃ Gmax : SimpleGraph V, G ≤ Gmax ∧ Gmax.IsMatchingFree ∧
      ∀ G', G' > Gmax → ∃ M : Subgraph G', M.IsPerfectMatching := by
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝ : Finite V
    h : G.IsMatchingFree
    ⊢ Exists fun Gmax => And (LE.le G Gmax) (And Gmax.IsMatchingFree (∀ (G' : Simp …
  -/
  simp_rw [← @not_forall_not _ Subgraph.IsPerfectMatching]
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝ : Finite V
    h : G.IsMatchingFree
    ⊢ Exists fun Gmax => And (LE.le G Gmax) (And Gmax.IsMatchingFree (∀ (G' : Simp …
  -/
  obtain ⟨Gmax, hGmax⟩ := Finite.exists_le_maximal h
  /-
    case intro
    V : Type u_1
    G : SimpleGraph V
    inst✝ : Finite V
    h : G.IsMatchingFree
    Gmax : SimpleGraph V
    hGmax : And (LE.le G Gmax) (Maximal SimpleGraph.IsMatchingFree Gmax)
    ⊢ Exists fun Gmax => And (LE.le G Gmax) (And Gmax.IsMatchingFree (∀ (G' : Simp …
  -/
  exact ⟨Gmax, ⟨hGmax.1, ⟨hGmax.2.prop, fun _ h' ↦ hGmax.2.not_prop_of_gt h'⟩⟩⟩
  /-
    🎉 no goals
  -/


/-- A graph `G` consists of a set of cycles, if each vertex is either isolated or connected to
exactly two vertices. This is used to create new matchings by taking the `symmDiff` with cycles.
The definition of `symmDiff` that makes sense is the one for `SimpleGraph`. The `symmDiff`
for `SimpleGraph.Subgraph` deriving from the lattice structure also affects the vertices included,
which we do not want in this case. This is why this property is defined for `SimpleGraph`, rather
than `SimpleGraph.Subgraph`.
-/
def IsCycles (G : SimpleGraph V) := ∀ ⦃v⦄, (G.neighborSet v).Nonempty → (G.neighborSet v).ncard = 2


/--
Given a vertex with one edge in a graph of cycles this gives the other edge incident
to the same vertex.
-/
lemma IsCycles.other_adj_of_adj (h : G.IsCycles) (hadj : G.Adj v w) :
    ∃ w', w ≠ w' ∧ G.Adj v w' := by
  /-
    V : Type u_1
    G : SimpleGraph V
    v w : V
    h : G.IsCycles
    hadj : G.Adj v w
    ⊢ Exists fun w' => And (Ne w w') (G.Adj v w')
  -/
  simp_rw [← SimpleGraph.mem_neighborSet] at hadj ⊢
  /-
    V : Type u_1
    G : SimpleGraph V
    v w : V
    h : G.IsCycles
    hadj : Membership.mem (G.neighborSet v) w
    ⊢ Exists fun w' => And (Ne w w') (Membership.mem (G.neighborSet v) w')
  -/
  have := h ⟨w, hadj⟩
  /-
    V : Type u_1
    G : SimpleGraph V
    v w : V
    h : G.IsCycles
    hadj : Membership.mem (G.neighborSet v) w
    this : Eq (G.neighborSet v).ncard 2
    ⊢ Exists fun w' => And (Ne w w') (Membership.mem (G.neighborSet v) w')
  -/
  obtain ⟨w', hww'⟩ := (G.neighborSet v).exists_ne_of_one_lt_ncard (by omega) w
  /-
    case intro
    V : Type u_1
    G : SimpleGraph V
    v w : V
    h : G.IsCycles
    hadj : Membership.mem (G.neighborSet v) w
    this : Eq (G.neighborSet v).ncard 2
    w' : V
    hww' : And (Membership.mem (G.neighborSet v) w') (Ne w' w)
    ⊢ Exists fun w' => And (Ne w w') (Membership.mem (G.neighborSet v) w')
  -/
  exact ⟨w', ⟨hww'.2.symm, hww'.1⟩⟩
  /-
    🎉 no goals
  -/


lemma Subgraph.IsPerfectMatching.symmDiff_spanningCoe_IsCycles
    {M : Subgraph G} {M' : Subgraph G'} (hM : M.IsPerfectMatching)
    (hM' : M'.IsPerfectMatching) : (M.spanningCoe ∆ M'.spanningCoe).IsCycles := by
  /-
    V : Type u_1
    G G' : SimpleGraph V
    M : G.Subgraph
    M' : G'.Subgraph
    hM : M.IsPerfectMatching
    hM' : M'.IsPerfectMatching
    ⊢ (symmDiff M.spanningCoe M'.spanningCoe).IsCycles
  -/
  intro v
  /-
    V : Type u_1
    G G' : SimpleGraph V
    M : G.Subgraph
    M' : G'.Subgraph
    hM : M.IsPerfectMatching
    hM' : M'.IsPerfectMatching
    v : V
    ⊢ ((symmDiff M.spanningCoe M'.spanningCoe).neighborSet v).Nonempty → Eq ((symm …
  -/
  obtain ⟨w, hw⟩ := hM.1 (hM.2 v)
  /-
    case intro
    V : Type u_1
    G G' : SimpleGraph V
    M : G.Subgraph
    M' : G'.Subgraph
    hM : M.IsPerfectMatching
    hM' : M'.IsPerfectMatching
    v w : V
    hw : And ((fun w => M.Adj v w) w) (∀ (y : V), (fun w => M.Adj v w) y → Eq y w)
    ⊢ ((symmDiff M.spanningCoe M'.spanningCoe).neighborSet v).Nonempty → Eq ((symm …
  -/
  obtain ⟨w', hw'⟩ := hM'.1 (hM'.2 v)
  simp only [symmDiff_def, Set.ncard_eq_two, ne_eq, imp_iff_not_or, Set.not_nonempty_iff_eq_empty,
    Set.eq_empty_iff_forall_not_mem, SimpleGraph.mem_neighborSet, SimpleGraph.sup_adj, sdiff_adj,
    spanningCoe_adj, not_or, not_and, not_not]
  /-
    case intro.intro
    V : Type u_1
    G G' : SimpleGraph V
    M : G.Subgraph
    M' : G'.Subgraph
    hM : M.IsPerfectMatching
    hM' : M'.IsPerfectMatching
    v w : V
    hw : And ((fun w => M.Adj v w) w) (∀ (y : V), (fun w => M.Adj v w) y → Eq y w)
    w' : V
    hw' : And ((fun w => M'.Adj v w) w') (∀ (y : V), (fun w => M'.Adj v w) y → Eq  …
    ⊢ Or (∀ (x : V), And (Or (Not (M.Adj v x)) (M'.Adj v x)) (Or (Not (M'.Adj v x) …
  -/
  by_cases hww' : w = w'
    /-
      case pos
      V : Type u_1
      G G' : SimpleGraph V
      M : G.Subgraph
      M' : G'.Subgraph
      hM : M.IsPerfectMatching
      hM' : M'.IsPerfectMatching
      v w : V
      hw : And ((fun w => M.Adj v w) w) (∀ (y : V), (fun w => M.Adj v w) y → Eq y w)
      w' : V
      hw' : And ((fun w => M'.Adj v w) w') (∀ (y : V), (fun w => M'.Adj v w) y → Eq  …
      hww' : Eq w w'
      ⊢ Or (∀ (x : V), And (Or (Not (M.Adj v x)) (M'.Adj v x)) (Or (Not (M'.Adj v x) …
    -/
  · simp_all [← imp_iff_not_or, hww']
    /-
      🎉 no goals
    -/
    /-
      case neg
      V : Type u_1
      G G' : SimpleGraph V
      M : G.Subgraph
      M' : G'.Subgraph
      hM : M.IsPerfectMatching
      hM' : M'.IsPerfectMatching
      v w : V
      hw : And ((fun w => M.Adj v w) w) (∀ (y : V), (fun w => M.Adj v w) y → Eq y w)
      w' : V
      hw' : And ((fun w => M'.Adj v w) w') (∀ (y : V), (fun w => M'.Adj v w) y → Eq  …
      hww' : Not (Eq w w')
      ⊢ Or (∀ (x : V), And (Or (Not (M.Adj v x)) (M'.Adj v x)) (Or (Not (M'.Adj v x) …
    -/
  · right
    /-
      case neg.h
      V : Type u_1
      G G' : SimpleGraph V
      M : G.Subgraph
      M' : G'.Subgraph
      hM : M.IsPerfectMatching
      hM' : M'.IsPerfectMatching
      v w : V
      hw : And ((fun w => M.Adj v w) w) (∀ (y : V), (fun w => M.Adj v w) y → Eq y w)
      w' : V
      hw' : And ((fun w => M'.Adj v w) w') (∀ (y : V), (fun w => M'.Adj v w) y → Eq  …
      hww' : Not (Eq w w')
      ⊢ Exists fun x => Exists fun y => And (Not (Eq x y)) (Eq ((Max.max (SDiff.sdif …
    -/
    use w, w'
    /-
      case h
      V : Type u_1
      G G' : SimpleGraph V
      M : G.Subgraph
      M' : G'.Subgraph
      hM : M.IsPerfectMatching
      hM' : M'.IsPerfectMatching
      v w : V
      hw : And ((fun w => M.Adj v w) w) (∀ (y : V), (fun w => M.Adj v w) y → Eq y w)
      w' : V
      hw' : And ((fun w => M'.Adj v w) w') (∀ (y : V), (fun w => M'.Adj v w) y → Eq  …
      hww' : Not (Eq w w')
      ⊢ And (Not (Eq w w')) (Eq ((Max.max (SDiff.sdiff M.spanningCoe M'.spanningCoe) …
    -/
    aesop
    /-
      🎉 no goals
    -/


/--
A graph `G` is alternating with respect to some other graph `G'`, if exactly every other edge in
`G` is in `G'`. Note that the degree of each vertex needs to be at most 2 for this to be
possible. This property is used to create new matchings using `symmDiff`.
The definition of `symmDiff` that makes sense is the one for `SimpleGraph`. The `symmDiff`
for `SimpleGraph.Subgraph` deriving from the lattice structure also affects the vertices included,
which we do not want in this case. This is why this property, just like `IsCycles`, is defined
for `SimpleGraph` rather than `SimpleGraph.Subgraph`.
-/
def IsAlternating (G G' : SimpleGraph V) :=
  ∀ ⦃v w w': V⦄, w ≠ w' → G.Adj v w → G.Adj v w' → (G'.Adj v w ↔ ¬ G'.Adj v w')


lemma IsPerfectMatching.symmDiff_spanningCoe_of_isAlternating {M : Subgraph G}
    (hM : M.IsPerfectMatching) (hG' : G'.IsAlternating M.spanningCoe) (hG'cyc : G'.IsCycles)  :
    (SimpleGraph.toSubgraph (M.spanningCoe ∆ G')
          /-
            V : Type u_1
            W : Type u_2
            G G' : SimpleGraph V
            M✝ M' : G.Subgraph
            v w : V
            M : G.Subgraph
            hM : M.IsPerfectMatching
            hG' : G'.IsAlternating M.spanningCoe
            hG'cyc : G'.IsCycles
            ⊢ LE.le (symmDiff M.spanningCoe G') ?m.156258
          -/
      (by rfl)).IsPerfectMatching := by
          /-
            🎉 no goals
          -/
  /-
    V : Type u_1
    G G' : SimpleGraph V
    M : G.Subgraph
    hM : M.IsPerfectMatching
    hG' : G'.IsAlternating M.spanningCoe
    hG'cyc : G'.IsCycles
    ⊢ (SimpleGraph.toSubgraph (symmDiff M.spanningCoe G') ⋯).IsPerfectMatching
  -/
  rw [Subgraph.isPerfectMatching_iff]
  /-
    V : Type u_1
    G G' : SimpleGraph V
    M : G.Subgraph
    hM : M.IsPerfectMatching
    hG' : G'.IsAlternating M.spanningCoe
    hG'cyc : G'.IsCycles
    ⊢ ∀ (v : V), ExistsUnique fun w => (SimpleGraph.toSubgraph (symmDiff M.spannin …
  -/
  intro v
  /-
    V : Type u_1
    G G' : SimpleGraph V
    M : G.Subgraph
    hM : M.IsPerfectMatching
    hG' : G'.IsAlternating M.spanningCoe
    hG'cyc : G'.IsCycles
    v : V
    ⊢ ExistsUnique fun w => (SimpleGraph.toSubgraph (symmDiff M.spanningCoe G') ⋯) …
  -/
  simp only [toSubgraph_adj, symmDiff_def, sup_adj, sdiff_adj, Subgraph.spanningCoe_adj]
  /-
    V : Type u_1
    G G' : SimpleGraph V
    M : G.Subgraph
    hM : M.IsPerfectMatching
    hG' : G'.IsAlternating M.spanningCoe
    hG'cyc : G'.IsCycles
    v : V
    ⊢ ExistsUnique fun w => Or (And (M.Adj v w) (Not (G'.Adj v w))) (And (G'.Adj v …
  -/
  obtain ⟨w, hw⟩ := hM.1 (hM.2 v)
  /-
    case intro
    V : Type u_1
    G G' : SimpleGraph V
    M : G.Subgraph
    hM : M.IsPerfectMatching
    hG' : G'.IsAlternating M.spanningCoe
    hG'cyc : G'.IsCycles
    v w : V
    hw : And ((fun w => M.Adj v w) w) (∀ (y : V), (fun w => M.Adj v w) y → Eq y w)
    ⊢ ExistsUnique fun w => Or (And (M.Adj v w) (Not (G'.Adj v w))) (And (G'.Adj v …
  -/
  by_cases h : G'.Adj v w
    /-
      case pos
      V : Type u_1
      G G' : SimpleGraph V
      M : G.Subgraph
      hM : M.IsPerfectMatching
      hG' : G'.IsAlternating M.spanningCoe
      hG'cyc : G'.IsCycles
      v w : V
      hw : And ((fun w => M.Adj v w) w) (∀ (y : V), (fun w => M.Adj v w) y → Eq y w)
      h : G'.Adj v w
      ⊢ ExistsUnique fun w => Or (And (M.Adj v w) (Not (G'.Adj v w))) (And (G'.Adj v …
    -/
  · obtain ⟨w', hw'⟩ := hG'cyc.other_adj_of_adj h
    /-
      case pos.intro
      V : Type u_1
      G G' : SimpleGraph V
      M : G.Subgraph
      hM : M.IsPerfectMatching
      hG' : G'.IsAlternating M.spanningCoe
      hG'cyc : G'.IsCycles
      v w : V
      hw : And ((fun w => M.Adj v w) w) (∀ (y : V), (fun w => M.Adj v w) y → Eq y w)
      h : G'.Adj v w
      w' : V
      hw' : And (Ne w w') (G'.Adj v w')
      ⊢ ExistsUnique fun w => Or (And (M.Adj v w) (Not (G'.Adj v w))) (And (G'.Adj v …
    -/
    have hmadj :  M.Adj v w ↔ ¬M.Adj v w' := by simpa using hG' hw'.1 h hw'.2
    /-
      case pos.intro
      V : Type u_1
      G G' : SimpleGraph V
      M : G.Subgraph
      hM : M.IsPerfectMatching
      hG' : G'.IsAlternating M.spanningCoe
      hG'cyc : G'.IsCycles
      v w : V
      hw : And ((fun w => M.Adj v w) w) (∀ (y : V), (fun w => M.Adj v w) y → Eq y w)
      h : G'.Adj v w
      w' : V
      hw' : And (Ne w w') (G'.Adj v w')
      hmadj : Iff (M.Adj v w) (Not (M.Adj v w'))
      ⊢ ExistsUnique fun w => Or (And (M.Adj v w) (Not (G'.Adj v w))) (And (G'.Adj v …
    -/
    use w'
    simp only [hmadj.mp hw.1, hw'.2, not_true_eq_false, and_self, not_false_eq_true, or_true,
      true_and]
    /-
      case h
      V : Type u_1
      G G' : SimpleGraph V
      M : G.Subgraph
      hM : M.IsPerfectMatching
      hG' : G'.IsAlternating M.spanningCoe
      hG'cyc : G'.IsCycles
      v w : V
      hw : And ((fun w => M.Adj v w) w) (∀ (y : V), (fun w => M.Adj v w) y → Eq y w)
      h : G'.Adj v w
      w' : V
      hw' : And (Ne w w') (G'.Adj v w')
      hmadj : Iff (M.Adj v w) (Not (M.Adj v w'))
      ⊢ ∀ (y : V), Or (And (M.Adj v y) (Not (G'.Adj v y))) (And (G'.Adj v y) (Not (M …
    -/
    rintro y (hl | hr)
      /-
        case h.inl
        V : Type u_1
        G G' : SimpleGraph V
        M : G.Subgraph
        hM : M.IsPerfectMatching
        hG' : G'.IsAlternating M.spanningCoe
        hG'cyc : G'.IsCycles
        v w : V
        hw : And ((fun w => M.Adj v w) w) (∀ (y : V), (fun w => M.Adj v w) y → Eq y w)
        h : G'.Adj v w
        w' : V
        hw' : And (Ne w w') (G'.Adj v w')
        hmadj : Iff (M.Adj v w) (Not (M.Adj v w'))
        y : V
        hl : And (M.Adj v y) (Not (G'.Adj v y))
        ⊢ Eq y w'
      -/
    · aesop
      /-
        🎉 no goals
      -/
      /-
        case h.inr
        V : Type u_1
        G G' : SimpleGraph V
        M : G.Subgraph
        hM : M.IsPerfectMatching
        hG' : G'.IsAlternating M.spanningCoe
        hG'cyc : G'.IsCycles
        v w : V
        hw : And ((fun w => M.Adj v w) w) (∀ (y : V), (fun w => M.Adj v w) y → Eq y w)
        h : G'.Adj v w
        w' : V
        hw' : And (Ne w w') (G'.Adj v w')
        hmadj : Iff (M.Adj v w) (Not (M.Adj v w'))
        y : V
        hr : And (G'.Adj v y) (Not (M.Adj v y))
        ⊢ Eq y w'
      -/
    · obtain ⟨w'', hw''⟩ := hG'cyc.other_adj_of_adj hr.1
      /-
        case h.inr.intro
        V : Type u_1
        G G' : SimpleGraph V
        M : G.Subgraph
        hM : M.IsPerfectMatching
        hG' : G'.IsAlternating M.spanningCoe
        hG'cyc : G'.IsCycles
        v w : V
        hw : And ((fun w => M.Adj v w) w) (∀ (y : V), (fun w => M.Adj v w) y → Eq y w)
        h : G'.Adj v w
        w' : V
        hw' : And (Ne w w') (G'.Adj v w')
        hmadj : Iff (M.Adj v w) (Not (M.Adj v w'))
        y : V
        hr : And (G'.Adj v y) (Not (M.Adj v y))
        w'' : V
        hw'' : And (Ne y w'') (G'.Adj v w'')
        ⊢ Eq y w'
      -/
      by_contra! hc
      simp_all only [show M.Adj v y ↔ ¬M.Adj v w' from by simpa using hG' hc hr.1 hw'.2,
        not_false_eq_true, ne_eq, iff_true, not_true_eq_false, and_false]
    /-
      case neg
      V : Type u_1
      G G' : SimpleGraph V
      M : G.Subgraph
      hM : M.IsPerfectMatching
      hG' : G'.IsAlternating M.spanningCoe
      hG'cyc : G'.IsCycles
      v w : V
      hw : And ((fun w => M.Adj v w) w) (∀ (y : V), (fun w => M.Adj v w) y → Eq y w)
      h : Not (G'.Adj v w)
      ⊢ ExistsUnique fun w => Or (And (M.Adj v w) (Not (G'.Adj v w))) (And (G'.Adj v …
    -/
  · use w
    /-
      case h
      V : Type u_1
      G G' : SimpleGraph V
      M : G.Subgraph
      hM : M.IsPerfectMatching
      hG' : G'.IsAlternating M.spanningCoe
      hG'cyc : G'.IsCycles
      v w : V
      hw : And ((fun w => M.Adj v w) w) (∀ (y : V), (fun w => M.Adj v w) y → Eq y w)
      h : Not (G'.Adj v w)
      ⊢ And ((fun w => Or (And (M.Adj v w) (Not (G'.Adj v w))) (And (G'.Adj v w) (No …
    -/
    simp only [hw.1, h, not_false_eq_true, and_self, not_true_eq_false, or_false, true_and]
    /-
      case h
      V : Type u_1
      G G' : SimpleGraph V
      M : G.Subgraph
      hM : M.IsPerfectMatching
      hG' : G'.IsAlternating M.spanningCoe
      hG'cyc : G'.IsCycles
      v w : V
      hw : And ((fun w => M.Adj v w) w) (∀ (y : V), (fun w => M.Adj v w) y → Eq y w)
      h : Not (G'.Adj v w)
      ⊢ ∀ (y : V), Or (And (M.Adj v y) (Not (G'.Adj v y))) (And (G'.Adj v y) (Not (M …
    -/
    rintro y (hl | hr)
      /-
        case h.inl
        V : Type u_1
        G G' : SimpleGraph V
        M : G.Subgraph
        hM : M.IsPerfectMatching
        hG' : G'.IsAlternating M.spanningCoe
        hG'cyc : G'.IsCycles
        v w : V
        hw : And ((fun w => M.Adj v w) w) (∀ (y : V), (fun w => M.Adj v w) y → Eq y w)
        h : Not (G'.Adj v w)
        y : V
        hl : And (M.Adj v y) (Not (G'.Adj v y))
        ⊢ Eq y w
      -/
    · exact hw.2 _ hl.1
      /-
        🎉 no goals
      -/
      /-
        case h.inr
        V : Type u_1
        G G' : SimpleGraph V
        M : G.Subgraph
        hM : M.IsPerfectMatching
        hG' : G'.IsAlternating M.spanningCoe
        hG'cyc : G'.IsCycles
        v w : V
        hw : And ((fun w => M.Adj v w) w) (∀ (y : V), (fun w => M.Adj v w) y → Eq y w)
        h : Not (G'.Adj v w)
        y : V
        hr : And (G'.Adj v y) (Not (M.Adj v y))
        ⊢ Eq y w
      -/
    · have ⟨w', hw'⟩ := hG'cyc.other_adj_of_adj hr.1
      simp_all only [show M.Adj v y ↔ ¬M.Adj v w' from by simpa using hG' hw'.1 hr.1 hw'.2, not_not,
        ne_eq, and_false]


