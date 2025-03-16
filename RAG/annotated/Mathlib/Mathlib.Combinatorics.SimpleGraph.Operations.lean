include f in
theorem card_edgeFinset_eq [Fintype G.edgeSet] [Fintype G'.edgeSet] :
    #G.edgeFinset = #G'.edgeFinset := by
  /-
    V : Type u_1
    G : SimpleGraph V
    W : Type u_2
    G' : SimpleGraph W
    f : G.Iso G'
    inst✝¹ : Fintype ↑G.edgeSet
    inst✝ : Fintype ↑G'.edgeSet
    ⊢ Eq G.edgeFinset.card G'.edgeFinset.card
  -/
  apply Finset.card_eq_of_equiv
  /-
    case i
    V : Type u_1
    G : SimpleGraph V
    W : Type u_2
    G' : SimpleGraph W
    f : G.Iso G'
    inst✝¹ : Fintype ↑G.edgeSet
    inst✝ : Fintype ↑G'.edgeSet
    ⊢ Equiv (Subtype fun x => Membership.mem G.edgeFinset x) (Subtype fun x => Mem …
  -/
  simp only [Set.mem_toFinset]
  /-
    case i
    V : Type u_1
    G : SimpleGraph V
    W : Type u_2
    G' : SimpleGraph W
    f : G.Iso G'
    inst✝¹ : Fintype ↑G.edgeSet
    inst✝ : Fintype ↑G'.edgeSet
    ⊢ Equiv (Subtype fun x => Membership.mem G.edgeSet x) (Subtype fun x => Member …
  -/
  exact f.mapEdgeSet
  /-
    🎉 no goals
  -/


/-- The graph formed by forgetting `t`'s neighbours and instead giving it those of `s`. The `s-t`
edge is removed if present. -/
def replaceVertex : SimpleGraph V where
  Adj v w := if v = t then if w = t then False else G.Adj s w
                      else if w = t then G.Adj v s else G.Adj v w
                 /-
                   V : Type u_1
                   G : SimpleGraph V
                   s t : V
                   inst✝ : DecidableEq V
                   v w : V
                   ⊢ (fun v w => ite (Eq v t) (ite (Eq w t) False (G.Adj s w)) (ite (Eq w t) (G.A …
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
  symm v w := by dsimp only; split_ifs <;> simp [adj_comm]
                                           /-
                                             🎉 no goals
                                           -/


/-- There is never an `s-t` edge in `G.replaceVertex s t`. -/
                                                                        /-
                                                                          V : Type u_1
                                                                          G : SimpleGraph V
                                                                          s t : V
                                                                          inst✝ : DecidableEq V
                                                                          ⊢ Not ((G.replaceVertex s t).Adj s t)
                                                                        -/
lemma not_adj_replaceVertex_same : ¬(G.replaceVertex s t).Adj s t := by simp [replaceVertex]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[simp] lemma replaceVertex_self : G.replaceVertex s s = G := by
  /-
    V : Type u_1
    G : SimpleGraph V
    s : V
    inst✝ : DecidableEq V
    ⊢ Eq (G.replaceVertex s s) G
  -/
  ext; unfold replaceVertex; aesop (add simp or_iff_not_imp_left)
                             /-
                               🎉 no goals
                             -/


/-- Except possibly for `t`, the neighbours of `s` in `G.replaceVertex s t` are its neighbours in
`G`. -/
lemma adj_replaceVertex_iff_of_ne_left {w : V} (hw : w ≠ t) :
                                                    /-
                                                      V : Type u_1
                                                      G : SimpleGraph V
                                                      s t : V
                                                      inst✝ : DecidableEq V
                                                      w : V
                                                      hw : Ne w t
                                                      ⊢ Iff ((G.replaceVertex s t).Adj s w) (G.Adj s w)
                                                    -/
    (G.replaceVertex s t).Adj s w ↔ G.Adj s w := by simp [replaceVertex, hw]
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- Except possibly for itself, the neighbours of `t` in `G.replaceVertex s t` are the neighbours of
`s` in `G`. -/
lemma adj_replaceVertex_iff_of_ne_right {w : V} (hw : w ≠ t) :
                                                    /-
                                                      V : Type u_1
                                                      G : SimpleGraph V
                                                      s t : V
                                                      inst✝ : DecidableEq V
                                                      w : V
                                                      hw : Ne w t
                                                      ⊢ Iff ((G.replaceVertex s t).Adj t w) (G.Adj s w)
                                                    -/
    (G.replaceVertex s t).Adj t w ↔ G.Adj s w := by simp [replaceVertex, hw]
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- Adjacency in `G.replaceVertex s t` which does not involve `t` is the same as that of `G`. -/
lemma adj_replaceVertex_iff_of_ne {v w : V} (hv : v ≠ t) (hw : w ≠ t) :
                                                    /-
                                                      V : Type u_1
                                                      G : SimpleGraph V
                                                      s t : V
                                                      inst✝ : DecidableEq V
                                                      v w : V
                                                      hv : Ne v t
                                                      hw : Ne w t
                                                      ⊢ Iff ((G.replaceVertex s t).Adj v w) (G.Adj v w)
                                                    -/
    (G.replaceVertex s t).Adj v w ↔ G.Adj v w := by simp [replaceVertex, hv, hw]
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem edgeSet_replaceVertex_of_not_adj (hn : ¬G.Adj s t) : (G.replaceVertex s t).edgeSet =
    G.edgeSet \ G.incidenceSet t ∪ (s(·, t)) '' (G.neighborSet s) := by
  /-
    V : Type u_1
    G : SimpleGraph V
    s t : V
    inst✝ : DecidableEq V
    hn : Not (G.Adj s t)
    ⊢ Eq (G.replaceVertex s t).edgeSet (Union.union (SDiff.sdiff G.edgeSet (G.inci …
  -/
  ext e; refine e.inductionOn ?_
  /-
    case h
    V : Type u_1
    G : SimpleGraph V
    s t : V
    inst✝ : DecidableEq V
    hn : Not (G.Adj s t)
    e : Sym2 V
    ⊢ ∀ (x y : V), Iff (Membership.mem (G.replaceVertex s t).edgeSet (Sym2.mk { fs …
  -/
  simp only [replaceVertex, mem_edgeSet, Set.mem_union, Set.mem_diff, mk'_mem_incidenceSet_iff]
  /-
    case h
    V : Type u_1
    G : SimpleGraph V
    s t : V
    inst✝ : DecidableEq V
    hn : Not (G.Adj s t)
    e : Sym2 V
    ⊢ ∀ (x y : V), Iff (ite (Eq x t) (ite (Eq y t) False (G.Adj s y)) (ite (Eq y t …
  -/
  intros; split_ifs; exacts [by simp_all, by aesop, by rw [adj_comm]; aesop, by aesop]
                     /-
                       🎉 no goals
                     -/


theorem edgeSet_replaceVertex_of_adj (ha : G.Adj s t) : (G.replaceVertex s t).edgeSet =
    (G.edgeSet \ G.incidenceSet t ∪ (s(·, t)) '' (G.neighborSet s)) \ {s(t, t)} := by
  /-
    V : Type u_1
    G : SimpleGraph V
    s t : V
    inst✝ : DecidableEq V
    ha : G.Adj s t
    ⊢ Eq (G.replaceVertex s t).edgeSet (SDiff.sdiff (Union.union (SDiff.sdiff G.ed …
  -/
  ext e; refine e.inductionOn ?_
  /-
    case h
    V : Type u_1
    G : SimpleGraph V
    s t : V
    inst✝ : DecidableEq V
    ha : G.Adj s t
    e : Sym2 V
    ⊢ ∀ (x y : V), Iff (Membership.mem (G.replaceVertex s t).edgeSet (Sym2.mk { fs …
  -/
  simp only [replaceVertex, mem_edgeSet, Set.mem_union, Set.mem_diff, mk'_mem_incidenceSet_iff]
  /-
    case h
    V : Type u_1
    G : SimpleGraph V
    s t : V
    inst✝ : DecidableEq V
    ha : G.Adj s t
    e : Sym2 V
    ⊢ ∀ (x y : V), Iff (ite (Eq x t) (ite (Eq y t) False (G.Adj s y)) (ite (Eq y t …
  -/
  intros; split_ifs; exacts [by simp_all, by aesop, by rw [adj_comm]; aesop, by aesop]
                     /-
                       🎉 no goals
                     -/


                                                        /-
                                                          V : Type u_1
                                                          G : SimpleGraph V
                                                          s t : V
                                                          inst✝² : DecidableEq V
                                                          inst✝¹ : Fintype V
                                                          inst✝ : DecidableRel G.Adj
                                                          ⊢ DecidableRel (G.replaceVertex s t).Adj
                                                        -/
instance : DecidableRel (G.replaceVertex s t).Adj := by unfold replaceVertex; infer_instance
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


theorem edgeFinset_replaceVertex_of_not_adj (hn : ¬G.Adj s t) : (G.replaceVertex s t).edgeFinset =
    G.edgeFinset \ G.incidenceFinset t ∪ (G.neighborFinset s).image (s(·, t)) := by
  simp only [incidenceFinset, neighborFinset, ← Set.toFinset_diff, ← Set.toFinset_image,
    ← Set.toFinset_union]
  /-
    V : Type u_1
    G : SimpleGraph V
    s t : V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    hn : Not (G.Adj s t)
    ⊢ Eq (G.replaceVertex s t).edgeFinset (Union.union (SDiff.sdiff G.edgeSet (G.i …
  -/
  exact Set.toFinset_congr (G.edgeSet_replaceVertex_of_not_adj hn)
  /-
    🎉 no goals
  -/


theorem edgeFinset_replaceVertex_of_adj (ha : G.Adj s t) : (G.replaceVertex s t).edgeFinset =
    (G.edgeFinset \ G.incidenceFinset t ∪ (G.neighborFinset s).image (s(·, t))) \ {s(t, t)} := by
  simp only [incidenceFinset, neighborFinset, ← Set.toFinset_diff, ← Set.toFinset_image,
    ← Set.toFinset_union, ← Set.toFinset_singleton]
  /-
    V : Type u_1
    G : SimpleGraph V
    s t : V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    ha : G.Adj s t
    ⊢ Eq (G.replaceVertex s t).edgeFinset (SDiff.sdiff (Union.union (SDiff.sdiff G …
  -/
  exact Set.toFinset_congr (G.edgeSet_replaceVertex_of_adj ha)
  /-
    🎉 no goals
  -/


lemma disjoint_sdiff_neighborFinset_image :
    Disjoint (G.edgeFinset \ G.incidenceFinset t) ((G.neighborFinset s).image (s(·, t))) := by
  /-
    V : Type u_1
    G : SimpleGraph V
    s t : V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    ⊢ Disjoint (SDiff.sdiff G.edgeFinset (G.incidenceFinset t)) (Finset.image (fun …
  -/
  rw [disjoint_iff_ne]
  /-
    V : Type u_1
    G : SimpleGraph V
    s t : V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    ⊢ ∀ (a : Sym2 V), Membership.mem (SDiff.sdiff G.edgeFinset (G.incidenceFinset  …
  -/
  intro e he
  have : t ∉ e := by
    rw [mem_sdiff, mem_incidenceFinset] at he
    obtain ⟨_, h⟩ := he
    contrapose! h
    simp_all [incidenceSet]
  /-
    V : Type u_1
    G : SimpleGraph V
    s t : V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    e : Sym2 V
    he : Membership.mem (SDiff.sdiff G.edgeFinset (G.incidenceFinset t)) e
    this : Not (Membership.mem e t)
    ⊢ ∀ (b : Sym2 V), Membership.mem (Finset.image (fun x => Sym2.mk { fst := x, s …
  -/
  aesop
  /-
    🎉 no goals
  -/


theorem card_edgeFinset_replaceVertex_of_not_adj (hn : ¬G.Adj s t) :
    #(G.replaceVertex s t).edgeFinset = #G.edgeFinset + G.degree s - G.degree t := by
  /-
    V : Type u_1
    G : SimpleGraph V
    s t : V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    hn : Not (G.Adj s t)
    ⊢ Eq (G.replaceVertex s t).edgeFinset.card (HSub.hSub (HAdd.hAdd G.edgeFinset. …
  -/
  have inc : G.incidenceFinset t ⊆ G.edgeFinset := by simp [incidenceFinset, incidenceSet_subset]
  rw [G.edgeFinset_replaceVertex_of_not_adj hn,
    card_union_of_disjoint G.disjoint_sdiff_neighborFinset_image, card_sdiff inc,
    ← Nat.sub_add_comm <| card_le_card inc, card_incidenceFinset_eq_degree]
  /-
    V : Type u_1
    G : SimpleGraph V
    s t : V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    hn : Not (G.Adj s t)
    inc : HasSubset.Subset (G.incidenceFinset t) G.edgeFinset
    ⊢ Eq (HSub.hSub (HAdd.hAdd G.edgeFinset.card (Finset.image (fun x => Sym2.mk { …
  -/
  congr 2
  /-
    case e_a.e_a
    V : Type u_1
    G : SimpleGraph V
    s t : V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    hn : Not (G.Adj s t)
    inc : HasSubset.Subset (G.incidenceFinset t) G.edgeFinset
    ⊢ Eq (Finset.image (fun x => Sym2.mk { fst := x, snd := t }) (G.neighborFinset …
  -/
  rw [card_image_of_injective, card_neighborFinset_eq_degree]
  /-
    case e_a.e_a.H
    V : Type u_1
    G : SimpleGraph V
    s t : V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    hn : Not (G.Adj s t)
    inc : HasSubset.Subset (G.incidenceFinset t) G.edgeFinset
    ⊢ Function.Injective fun x => Sym2.mk { fst := x, snd := t }
  -/
  unfold Function.Injective
  /-
    case e_a.e_a.H
    V : Type u_1
    G : SimpleGraph V
    s t : V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    hn : Not (G.Adj s t)
    inc : HasSubset.Subset (G.incidenceFinset t) G.edgeFinset
    ⊢ ∀ ⦃a₁ a₂ : V⦄, Eq ((fun x => Sym2.mk { fst := x, snd := t }) a₁) ((fun x =>  …
  -/
  aesop
  /-
    🎉 no goals
  -/


theorem card_edgeFinset_replaceVertex_of_adj (ha : G.Adj s t) :
    #(G.replaceVertex s t).edgeFinset = #G.edgeFinset + G.degree s - G.degree t - 1 := by
  /-
    V : Type u_1
    G : SimpleGraph V
    s t : V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    ha : G.Adj s t
    ⊢ Eq (G.replaceVertex s t).edgeFinset.card (HSub.hSub (HSub.hSub (HAdd.hAdd G. …
  -/
  have inc : G.incidenceFinset t ⊆ G.edgeFinset := by simp [incidenceFinset, incidenceSet_subset]
  rw [G.edgeFinset_replaceVertex_of_adj ha, card_sdiff (by simp [ha]),
    card_union_of_disjoint G.disjoint_sdiff_neighborFinset_image, card_sdiff inc,
    ← Nat.sub_add_comm <| card_le_card inc, card_incidenceFinset_eq_degree]
  /-
    V : Type u_1
    G : SimpleGraph V
    s t : V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    ha : G.Adj s t
    inc : HasSubset.Subset (G.incidenceFinset t) G.edgeFinset
    ⊢ Eq (HSub.hSub (HSub.hSub (HAdd.hAdd G.edgeFinset.card (Finset.image (fun x = …
  -/
  congr 2
  /-
    case e_a.e_a
    V : Type u_1
    G : SimpleGraph V
    s t : V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    ha : G.Adj s t
    inc : HasSubset.Subset (G.incidenceFinset t) G.edgeFinset
    ⊢ Eq (HAdd.hAdd G.edgeFinset.card (Finset.image (fun x => Sym2.mk { fst := x,  …
  -/
  rw [card_image_of_injective, card_neighborFinset_eq_degree]
  /-
    case e_a.e_a.H
    V : Type u_1
    G : SimpleGraph V
    s t : V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    ha : G.Adj s t
    inc : HasSubset.Subset (G.incidenceFinset t) G.edgeFinset
    ⊢ Function.Injective fun x => Sym2.mk { fst := x, snd := t }
  -/
  unfold Function.Injective
  /-
    case e_a.e_a.H
    V : Type u_1
    G : SimpleGraph V
    s t : V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    ha : G.Adj s t
    inc : HasSubset.Subset (G.incidenceFinset t) G.edgeFinset
    ⊢ ∀ ⦃a₁ a₂ : V⦄, Eq ((fun x => Sym2.mk { fst := x, snd := t }) a₁) ((fun x =>  …
  -/
  aesop
  /-
    🎉 no goals
  -/


/-- The graph with a single `s-t` edge. It is empty iff `s = t`. -/
def edge : SimpleGraph V := fromEdgeSet {s(s, t)}


lemma edge_adj (v w : V) : (edge s t).Adj v w ↔ (v = s ∧ w = t ∨ v = t ∧ w = s) ∧ v ≠ w := by
  /-
    V : Type u_1
    s t v w : V
    ⊢ Iff ((SimpleGraph.edge s t).Adj v w) (And (Or (And (Eq v s) (Eq w t)) (And ( …
  -/
  rw [edge, fromEdgeSet_adj, Set.mem_singleton_iff, Sym2.eq_iff]
  /-
    🎉 no goals
  -/


variable [DecidableEq V] in
instance : DecidableRel (edge s t).Adj := fun _ _ ↦ by
  /-
    V : Type u_1
    G : SimpleGraph V
    s t : V
    inst✝ : DecidableEq V
    x✝¹ x✝ : V
    ⊢ Decidable ((SimpleGraph.edge s t).Adj x✝¹ x✝)
  -/
  rw [edge_adj]; infer_instance
                 /-
                   🎉 no goals
                 -/


lemma edge_self_eq_bot : edge s s = ⊥ := by
  /-
    V : Type u_1
    s : V
    ⊢ Eq (SimpleGraph.edge s s) Bot.bot
  -/
  ext; rw [edge_adj]; aesop
                      /-
                        🎉 no goals
                      -/


@[simp]
lemma sup_edge_self : G ⊔ edge s s = G := by
  /-
    V : Type u_1
    G : SimpleGraph V
    s : V
    ⊢ Eq (Max.max G (SimpleGraph.edge s s)) G
  -/
  rw [edge_self_eq_bot, sup_of_le_left bot_le]
  /-
    🎉 no goals
  -/


lemma edge_edgeSet_of_ne (h : s ≠ t) : (edge s t).edgeSet = {s(s, t)} := by
  rwa [edge, edgeSet_fromEdgeSet, sdiff_eq_left, Set.disjoint_singleton_left, Set.mem_setOf_eq,
    Sym2.isDiag_iff_proj_eq]


lemma sup_edge_of_adj (h : G.Adj s t) : G ⊔ edge s t = G := by
  rwa [sup_eq_left, ← edgeSet_subset_edgeSet, edge_edgeSet_of_ne h.ne, Set.singleton_subset_iff,
    mem_edgeSet]


variable [DecidableEq V] in
                                            /-
                                              V : Type u_1
                                              G : SimpleGraph V
                                              s t : V
                                              inst✝² : Fintype V
                                              inst✝¹ : DecidableRel G.Adj
                                              inst✝ : DecidableEq V
                                              ⊢ Fintype ↑(SimpleGraph.edge s t).edgeSet
                                            -/
instance : Fintype (edge s t).edgeSet := by rw [edge]; infer_instance
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem edgeFinset_sup_edge [Fintype (edgeSet (G ⊔ edge s t))] (hn : ¬G.Adj s t) (h : s ≠ t) :
                                                              /-
                                                                V : Type u_1
                                                                G : SimpleGraph V
                                                                s t : V
                                                                inst✝² : Fintype V
                                                                inst✝¹ : DecidableRel G.Adj
                                                                inst✝ : Fintype ↑(Max.max G (SimpleGraph.edge s t)).edgeSet
                                                                hn : Not (G.Adj s t)
                                                                h : Ne s t
                                                                ⊢ Not (Membership.mem G.edgeFinset (Sym2.mk { fst := s, snd := t }))
                                                              -/
    (G ⊔ edge s t).edgeFinset = G.edgeFinset.cons s(s, t) (by simp_all) := by
                                                              /-
                                                                🎉 no goals
                                                              -/
  /-
    V : Type u_1
    G : SimpleGraph V
    s t : V
    inst✝² : Fintype V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : Fintype ↑(Max.max G (SimpleGraph.edge s t)).edgeSet
    hn : Not (G.Adj s t)
    h : Ne s t
    ⊢ Eq (Max.max G (SimpleGraph.edge s t)).edgeFinset (Finset.cons (Sym2.mk { fst …
  -/
  letI := Classical.decEq V
  /-
    V : Type u_1
    G : SimpleGraph V
    s t : V
    inst✝² : Fintype V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : Fintype ↑(Max.max G (SimpleGraph.edge s t)).edgeSet
    hn : Not (G.Adj s t)
    h : Ne s t
    this : DecidableEq V := Classical.decEq V
    ⊢ Eq (Max.max G (SimpleGraph.edge s t)).edgeFinset (Finset.cons (Sym2.mk { fst …
  -/
  rw [edgeFinset_sup, cons_eq_insert, insert_eq, union_comm]
  /-
    V : Type u_1
    G : SimpleGraph V
    s t : V
    inst✝² : Fintype V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : Fintype ↑(Max.max G (SimpleGraph.edge s t)).edgeSet
    hn : Not (G.Adj s t)
    h : Ne s t
    this : DecidableEq V := Classical.decEq V
    ⊢ Eq (Union.union (SimpleGraph.edge s t).edgeFinset G.edgeFinset) (Union.union …
  -/
  simp_rw [edgeFinset, edge_edgeSet_of_ne h]; rfl
                                              /-
                                                🎉 no goals
                                              -/


theorem card_edgeFinset_sup_edge [Fintype (edgeSet (G ⊔ edge s t))] (hn : ¬G.Adj s t) (h : s ≠ t) :
    #(G ⊔ edge s t).edgeFinset = #G.edgeFinset + 1 := by
  /-
    V : Type u_1
    G : SimpleGraph V
    s t : V
    inst✝² : Fintype V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : Fintype ↑(Max.max G (SimpleGraph.edge s t)).edgeSet
    hn : Not (G.Adj s t)
    h : Ne s t
    ⊢ Eq (Max.max G (SimpleGraph.edge s t)).edgeFinset.card (HAdd.hAdd G.edgeFinse …
  -/
  rw [G.edgeFinset_sup_edge hn h, card_cons]
  /-
    🎉 no goals
  -/


