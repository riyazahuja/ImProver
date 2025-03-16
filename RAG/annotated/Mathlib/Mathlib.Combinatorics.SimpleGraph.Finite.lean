/-- The `edgeSet` of the graph as a `Finset`. -/
abbrev edgeFinset : Finset (Sym2 V) :=
  Set.toFinset G.edgeSet


@[norm_cast]
theorem coe_edgeFinset : (G.edgeFinset : Set (Sym2 V)) = G.edgeSet :=
  Set.coe_toFinset _


theorem mem_edgeFinset : e ∈ G.edgeFinset ↔ e ∈ G.edgeSet :=
  Set.mem_toFinset


theorem not_isDiag_of_mem_edgeFinset : e ∈ G.edgeFinset → ¬e.IsDiag :=
  not_isDiag_of_mem_edgeSet _ ∘ mem_edgeFinset.1


                                                                       /-
                                                                         V : Type u_1
                                                                         G₁ G₂ : SimpleGraph V
                                                                         inst✝¹ : Fintype ↑G₁.edgeSet
                                                                         inst✝ : Fintype ↑G₂.edgeSet
                                                                         ⊢ Iff (Eq G₁.edgeFinset G₂.edgeFinset) (Eq G₁ G₂)
                                                                       -/
theorem edgeFinset_inj : G₁.edgeFinset = G₂.edgeFinset ↔ G₁ = G₂ := by simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


                                                                                     /-
                                                                                       V : Type u_1
                                                                                       G₁ G₂ : SimpleGraph V
                                                                                       inst✝¹ : Fintype ↑G₁.edgeSet
                                                                                       inst✝ : Fintype ↑G₂.edgeSet
                                                                                       ⊢ Iff (HasSubset.Subset G₁.edgeFinset G₂.edgeFinset) (LE.le G₁ G₂)
                                                                                     -/
theorem edgeFinset_subset_edgeFinset : G₁.edgeFinset ⊆ G₂.edgeFinset ↔ G₁ ≤ G₂ := by simp
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


                                                                                      /-
                                                                                        V : Type u_1
                                                                                        G₁ G₂ : SimpleGraph V
                                                                                        inst✝¹ : Fintype ↑G₁.edgeSet
                                                                                        inst✝ : Fintype ↑G₂.edgeSet
                                                                                        ⊢ Iff (HasSSubset.SSubset G₁.edgeFinset G₂.edgeFinset) (LT.lt G₁ G₂)
                                                                                      -/
theorem edgeFinset_ssubset_edgeFinset : G₁.edgeFinset ⊂ G₂.edgeFinset ↔ G₁ < G₂ := by simp
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


@[gcongr] alias ⟨_, edgeFinset_mono⟩ := edgeFinset_subset_edgeFinset


alias ⟨_, edgeFinset_strict_mono⟩ := edgeFinset_ssubset_edgeFinset


@[simp]
                                                                  /-
                                                                    V : Type u_1
                                                                    ⊢ Eq Bot.bot.edgeFinset EmptyCollection.emptyCollection
                                                                  -/
theorem edgeFinset_bot : (⊥ : SimpleGraph V).edgeFinset = ∅ := by simp [edgeFinset]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp]
theorem edgeFinset_sup [Fintype (edgeSet (G₁ ⊔ G₂))] [DecidableEq V] :
                                                               /-
                                                                 V : Type u_1
                                                                 G₁ G₂ : SimpleGraph V
                                                                 inst✝³ : Fintype ↑G₁.edgeSet
                                                                 inst✝² : Fintype ↑G₂.edgeSet
                                                                 inst✝¹ : Fintype ↑(Max.max G₁ G₂).edgeSet
                                                                 inst✝ : DecidableEq V
                                                                 ⊢ Eq (Max.max G₁ G₂).edgeFinset (Union.union G₁.edgeFinset G₂.edgeFinset)
                                                               -/
    (G₁ ⊔ G₂).edgeFinset = G₁.edgeFinset ∪ G₂.edgeFinset := by simp [edgeFinset]
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[simp]
theorem edgeFinset_inf [DecidableEq V] : (G₁ ⊓ G₂).edgeFinset = G₁.edgeFinset ∩ G₂.edgeFinset := by
  /-
    V : Type u_1
    G₁ G₂ : SimpleGraph V
    inst✝² : Fintype ↑G₁.edgeSet
    inst✝¹ : Fintype ↑G₂.edgeSet
    inst✝ : DecidableEq V
    ⊢ Eq (Min.min G₁ G₂).edgeFinset (Inter.inter G₁.edgeFinset G₂.edgeFinset)
  -/
  simp [edgeFinset]
  /-
    🎉 no goals
  -/


@[simp]
theorem edgeFinset_sdiff [DecidableEq V] :
                                                               /-
                                                                 V : Type u_1
                                                                 G₁ G₂ : SimpleGraph V
                                                                 inst✝² : Fintype ↑G₁.edgeSet
                                                                 inst✝¹ : Fintype ↑G₂.edgeSet
                                                                 inst✝ : DecidableEq V
                                                                 ⊢ Eq (SDiff.sdiff G₁ G₂).edgeFinset (SDiff.sdiff G₁.edgeFinset G₂.edgeFinset)
                                                               -/
    (G₁ \ G₂).edgeFinset = G₁.edgeFinset \ G₂.edgeFinset := by simp [edgeFinset]
                                                               /-
                                                                 🎉 no goals
                                                               -/


lemma disjoint_edgeFinset : Disjoint G₁.edgeFinset G₂.edgeFinset ↔ Disjoint G₁ G₂ := by
  /-
    V : Type u_1
    G₁ G₂ : SimpleGraph V
    inst✝¹ : Fintype ↑G₁.edgeSet
    inst✝ : Fintype ↑G₂.edgeSet
    ⊢ Iff (Disjoint G₁.edgeFinset G₂.edgeFinset) (Disjoint G₁ G₂)
  -/
  simp_rw [← Finset.disjoint_coe, coe_edgeFinset, disjoint_edgeSet]
  /-
    🎉 no goals
  -/


lemma edgeFinset_eq_empty : G.edgeFinset = ∅ ↔ G = ⊥ := by
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝ : Fintype ↑G.edgeSet
    ⊢ Iff (Eq G.edgeFinset EmptyCollection.emptyCollection) (Eq G Bot.bot)
  -/
  rw [← edgeFinset_bot, edgeFinset_inj]
  /-
    🎉 no goals
  -/


lemma edgeFinset_nonempty : G.edgeFinset.Nonempty ↔ G ≠ ⊥ := by
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝ : Fintype ↑G.edgeSet
    ⊢ Iff G.edgeFinset.Nonempty (Ne G Bot.bot)
  -/
  rw [Finset.nonempty_iff_ne_empty, edgeFinset_eq_empty.ne]
  /-
    🎉 no goals
  -/


theorem edgeFinset_card : #G.edgeFinset = Fintype.card G.edgeSet :=
  Set.toFinset_card _


@[simp]
theorem edgeSet_univ_card : #(univ : Finset G.edgeSet) = #G.edgeFinset :=
  Fintype.card_of_subtype G.edgeFinset fun _ => mem_edgeFinset


@[simp]
theorem edgeFinset_top [DecidableEq V] :
                                                                        /-
                                                                          V : Type u_1
                                                                          inst✝¹ : Fintype V
                                                                          inst✝ : DecidableEq V
                                                                          ⊢ Eq Top.top.edgeFinset (Finset.filter (fun e => Not e.IsDiag) Finset.univ)
                                                                        -/
    (⊤ : SimpleGraph V).edgeFinset = ({e | ¬e.IsDiag} : Finset _) := by simp [← coe_inj]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


/-- The complete graph on `n` vertices has `n.choose 2` edges. -/
theorem card_edgeFinset_top_eq_card_choose_two [DecidableEq V] :
    #(⊤ : SimpleGraph V).edgeFinset = (Fintype.card V).choose 2 := by
  /-
    V : Type u_1
    inst✝¹ : Fintype V
    inst✝ : DecidableEq V
    ⊢ Eq Top.top.edgeFinset.card ((Fintype.card V).choose 2)
  -/
  simp_rw [Set.toFinset_card, edgeSet_top, Set.coe_setOf, ← Sym2.card_subtype_not_diag]
  /-
    🎉 no goals
  -/


/-- Any graph on `n` vertices has at most `n.choose 2` edges. -/
theorem card_edgeFinset_le_card_choose_two : #G.edgeFinset ≤ (Fintype.card V).choose 2 := by
  classical
  rw [← card_edgeFinset_top_eq_card_choose_two]
  exact card_le_card (edgeFinset_mono le_top)


theorem edgeFinset_deleteEdges [DecidableEq V] [Fintype G.edgeSet] (s : Finset (Sym2 V))
    [Fintype (G.deleteEdges s).edgeSet] :
    (G.deleteEdges s).edgeFinset = G.edgeFinset \ s := by
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype ↑G.edgeSet
    s : Finset (Sym2 V)
    inst✝ : Fintype ↑(G.deleteEdges ↑s).edgeSet
    ⊢ Eq (G.deleteEdges ↑s).edgeFinset (SDiff.sdiff G.edgeFinset s)
  -/
  ext e
  /-
    case h
    V : Type u_1
    G : SimpleGraph V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype ↑G.edgeSet
    s : Finset (Sym2 V)
    inst✝ : Fintype ↑(G.deleteEdges ↑s).edgeSet
    e : Sym2 V
    ⊢ Iff (Membership.mem (G.deleteEdges ↑s).edgeFinset e) (Membership.mem (SDiff. …
  -/
  simp [edgeSet_deleteEdges]
  /-
    🎉 no goals
  -/


/-- A graph is `r`-*delete-far* from a property `p` if we must delete at least `r` edges from it to
get a graph with the property `p`. -/
def DeleteFar (p : SimpleGraph V → Prop) (r : 𝕜) : Prop :=
  ∀ ⦃s⦄, s ⊆ G.edgeFinset → p (G.deleteEdges s) → r ≤ #s


theorem deleteFar_iff [Fintype (Sym2 V)] :
    G.DeleteFar p r ↔ ∀ ⦃H : SimpleGraph _⦄ [DecidableRel H.Adj],
      H ≤ G → p H → r ≤ #G.edgeFinset - #H.edgeFinset := by
  classical
  refine ⟨fun h H _ hHG hH ↦ ?_, fun h s hs hG ↦ ?_⟩
  · have := h (sdiff_subset (t := H.edgeFinset))
    simp only [deleteEdges_sdiff_eq_of_le hHG, edgeFinset_mono hHG, card_sdiff,
      card_le_card, coe_sdiff, coe_edgeFinset, Nat.cast_sub] at this
    exact this hH
  · classical
    simpa [card_sdiff hs, edgeFinset_deleteEdges, -Set.toFinset_card, Nat.cast_sub,
      card_le_card hs] using h (G.deleteEdges_le s) hG


alias ⟨DeleteFar.le_card_sub_card, _⟩ := deleteFar_iff


theorem DeleteFar.mono (h : G.DeleteFar p r₂) (hr : r₁ ≤ r₂) : G.DeleteFar p r₁ := fun _ hs hG =>
  hr.trans <| h hs hG


/-- `G.neighbors v` is the `Finset` version of `G.Adj v` in case `G` is
locally finite at `v`. -/
def neighborFinset : Finset V :=
  (G.neighborSet v).toFinset


theorem neighborFinset_def : G.neighborFinset v = (G.neighborSet v).toFinset :=
  rfl


@[simp]
theorem mem_neighborFinset (w : V) : w ∈ G.neighborFinset v ↔ G.Adj v w :=
  Set.mem_toFinset


                                                                   /-
                                                                     V : Type u_1
                                                                     G : SimpleGraph V
                                                                     v : V
                                                                     inst✝ : Fintype ↑(G.neighborSet v)
                                                                     ⊢ Not (Membership.mem (G.neighborFinset v) v)
                                                                   -/
theorem not_mem_neighborFinset_self : v ∉ G.neighborFinset v := by simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


theorem neighborFinset_disjoint_singleton : Disjoint (G.neighborFinset v) {v} :=
  Finset.disjoint_singleton_right.mpr <| not_mem_neighborFinset_self _ _


theorem singleton_disjoint_neighborFinset : Disjoint {v} (G.neighborFinset v) :=
  Finset.disjoint_singleton_left.mpr <| not_mem_neighborFinset_self _ _


/-- `G.degree v` is the number of vertices adjacent to `v`. -/
def degree : ℕ := #(G.neighborFinset v)

-- Porting note: in Lean 3 we could do `simp [← degree]`, but that gives
-- "invalid '←' modifier, 'SimpleGraph.degree' is a declaration name to be unfolded".
-- In any case, having this lemma is good since there's no guarantee we won't still change
-- the definition of `degree`.

@[simp]
theorem card_neighborFinset_eq_degree : #(G.neighborFinset v) = G.degree v := rfl


@[simp]
theorem card_neighborSet_eq_degree : Fintype.card (G.neighborSet v) = G.degree v :=
  (Set.toFinset_card _).symm


theorem degree_pos_iff_exists_adj : 0 < G.degree v ↔ ∃ w, G.Adj v w := by
  /-
    V : Type u_1
    G : SimpleGraph V
    v : V
    inst✝ : Fintype ↑(G.neighborSet v)
    ⊢ Iff (LT.lt 0 (G.degree v)) (Exists fun w => G.Adj v w)
  -/
  simp only [degree, card_pos, Finset.Nonempty, mem_neighborFinset]
  /-
    🎉 no goals
  -/


theorem degree_compl [Fintype (Gᶜ.neighborSet v)] [Fintype V] :
    Gᶜ.degree v = Fintype.card V - 1 - G.degree v := by
  classical
    rw [← card_neighborSet_union_compl_neighborSet G v, Set.toFinset_union]
    simp [card_union_of_disjoint (Set.disjoint_toFinset.mpr (compl_neighborSet_disjoint G v))]


instance incidenceSetFintype [DecidableEq V] : Fintype (G.incidenceSet v) :=
  Fintype.ofEquiv (G.neighborSet v) (G.incidenceSetEquivNeighborSet v).symm


/-- This is the `Finset` version of `incidenceSet`. -/
def incidenceFinset [DecidableEq V] : Finset (Sym2 V) :=
  (G.incidenceSet v).toFinset


@[simp]
theorem card_incidenceSet_eq_degree [DecidableEq V] :
    Fintype.card (G.incidenceSet v) = G.degree v := by
  /-
    V : Type u_1
    G : SimpleGraph V
    v : V
    inst✝¹ : Fintype ↑(G.neighborSet v)
    inst✝ : DecidableEq V
    ⊢ Eq (Fintype.card ↑(G.incidenceSet v)) (G.degree v)
  -/
  rw [Fintype.card_congr (G.incidenceSetEquivNeighborSet v)]
  /-
    V : Type u_1
    G : SimpleGraph V
    v : V
    inst✝¹ : Fintype ↑(G.neighborSet v)
    inst✝ : DecidableEq V
    ⊢ Eq (Fintype.card ↑(G.neighborSet v)) (G.degree v)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem card_incidenceFinset_eq_degree [DecidableEq V] : #(G.incidenceFinset v) = G.degree v := by
  /-
    V : Type u_1
    G : SimpleGraph V
    v : V
    inst✝¹ : Fintype ↑(G.neighborSet v)
    inst✝ : DecidableEq V
    ⊢ Eq (G.incidenceFinset v).card (G.degree v)
  -/
  rw [← G.card_incidenceSet_eq_degree]
  /-
    V : Type u_1
    G : SimpleGraph V
    v : V
    inst✝¹ : Fintype ↑(G.neighborSet v)
    inst✝ : DecidableEq V
    ⊢ Eq (G.incidenceFinset v).card (Fintype.card ↑(G.incidenceSet v))
  -/
  apply Set.toFinset_card
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_incidenceFinset [DecidableEq V] (e : Sym2 V) :
    e ∈ G.incidenceFinset v ↔ e ∈ G.incidenceSet v :=
  Set.mem_toFinset


theorem incidenceFinset_eq_filter [DecidableEq V] [Fintype G.edgeSet] :
    G.incidenceFinset v = {e ∈ G.edgeFinset | v ∈ e} := by
  /-
    V : Type u_1
    G : SimpleGraph V
    v : V
    inst✝² : Fintype ↑(G.neighborSet v)
    inst✝¹ : DecidableEq V
    inst✝ : Fintype ↑G.edgeSet
    ⊢ Eq (G.incidenceFinset v) (Finset.filter (fun e => Membership.mem e v) G.edge …
  -/
  ext e
  /-
    case h
    V : Type u_1
    G : SimpleGraph V
    v : V
    inst✝² : Fintype ↑(G.neighborSet v)
    inst✝¹ : DecidableEq V
    inst✝ : Fintype ↑G.edgeSet
    e : Sym2 V
    ⊢ Iff (Membership.mem (G.incidenceFinset v) e) (Membership.mem (Finset.filter  …
  -/
  induction e
  /-
    case h.h
    V : Type u_1
    G : SimpleGraph V
    v : V
    inst✝² : Fintype ↑(G.neighborSet v)
    inst✝¹ : DecidableEq V
    inst✝ : Fintype ↑G.edgeSet
    x✝ y✝ : V
    ⊢ Iff (Membership.mem (G.incidenceFinset v) (Sym2.mk { fst := x✝, snd := y✝ }) …
  -/
  simp [mk'_mem_incidenceSet_iff]
  /-
    🎉 no goals
  -/


/-- A graph is locally finite if every vertex has a finite neighbor set. -/
abbrev LocallyFinite :=
  ∀ v : V, Fintype (G.neighborSet v)


/-- A locally finite simple graph is regular of degree `d` if every vertex has degree `d`. -/
def IsRegularOfDegree (d : ℕ) : Prop :=
  ∀ v : V, G.degree v = d


theorem IsRegularOfDegree.degree_eq {d : ℕ} (h : G.IsRegularOfDegree d) (v : V) : G.degree v = d :=
  h v


theorem IsRegularOfDegree.compl [Fintype V] [DecidableEq V] {G : SimpleGraph V} [DecidableRel G.Adj]
    {k : ℕ} (h : G.IsRegularOfDegree k) : Gᶜ.IsRegularOfDegree (Fintype.card V - 1 - k) := by
  /-
    V : Type u_1
    inst✝² : Fintype V
    inst✝¹ : DecidableEq V
    G : SimpleGraph V
    inst✝ : DecidableRel G.Adj
    k : Nat
    h : G.IsRegularOfDegree k
    ⊢ (HasCompl.compl G).IsRegularOfDegree (HSub.hSub (HSub.hSub (Fintype.card V)  …
  -/
  intro v
  /-
    V : Type u_1
    inst✝² : Fintype V
    inst✝¹ : DecidableEq V
    G : SimpleGraph V
    inst✝ : DecidableRel G.Adj
    k : Nat
    h : G.IsRegularOfDegree k
    v : V
    ⊢ Eq ((HasCompl.compl G).degree v) (HSub.hSub (HSub.hSub (Fintype.card V) 1) k)
  -/
  rw [degree_compl, h v]
  /-
    🎉 no goals
  -/


instance neighborSetFintype [DecidableRel G.Adj] (v : V) : Fintype (G.neighborSet v) :=
  @Subtype.fintype _ (· ∈ G.neighborSet v)
    (by
      /-
        V : Type u_1
        G : SimpleGraph V
        e : Sym2 V
        inst✝¹ : Fintype V
        inst✝ : DecidableRel G.Adj
        v : V
        ⊢ DecidablePred fun x => Membership.mem (G.neighborSet v) x
      -/
      simp_rw [mem_neighborSet]
      /-
        V : Type u_1
        G : SimpleGraph V
        e : Sym2 V
        inst✝¹ : Fintype V
        inst✝ : DecidableRel G.Adj
        v : V
        ⊢ DecidablePred fun x => G.Adj v x
      -/
      infer_instance)
      /-
        🎉 no goals
      -/
    _


theorem neighborFinset_eq_filter {v : V} [DecidableRel G.Adj] :
                                                            /-
                                                              V : Type u_1
                                                              G : SimpleGraph V
                                                              inst✝¹ : Fintype V
                                                              v : V
                                                              inst✝ : DecidableRel G.Adj
                                                              ⊢ Eq (G.neighborFinset v) (Finset.filter (fun w => G.Adj v w) Finset.univ)
                                                            -/
    G.neighborFinset v = ({w | G.Adj v w} : Finset _) := by ext; simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem neighborFinset_compl [DecidableEq V] [DecidableRel G.Adj] (v : V) :
    Gᶜ.neighborFinset v = (G.neighborFinset v)ᶜ \ {v} := by
  simp only [neighborFinset, neighborSet_compl, Set.toFinset_diff, Set.toFinset_compl,
    Set.toFinset_singleton]


@[simp]
theorem complete_graph_degree [DecidableEq V] (v : V) :
    (⊤ : SimpleGraph V).degree v = Fintype.card V - 1 := by
  /-
    V : Type u_1
    inst✝¹ : Fintype V
    inst✝ : DecidableEq V
    v : V
    ⊢ Eq (Top.top.degree v) (HSub.hSub (Fintype.card V) 1)
  -/
  erw [degree, neighborFinset_eq_filter, filter_ne, card_erase_of_mem (mem_univ v), card_univ]
  /-
    🎉 no goals
  -/


theorem bot_degree (v : V) : (⊥ : SimpleGraph V).degree v = 0 := by
  /-
    V : Type u_1
    inst✝ : Fintype V
    v : V
    ⊢ Eq (Bot.bot.degree v) 0
  -/
  erw [degree, neighborFinset_eq_filter, filter_False]
  /-
    V : Type u_1
    inst✝ : Fintype V
    v : V
    ⊢ Eq EmptyCollection.emptyCollection.card 0
  -/
  exact Finset.card_empty
  /-
    🎉 no goals
  -/


theorem IsRegularOfDegree.top [DecidableEq V] :
    (⊤ : SimpleGraph V).IsRegularOfDegree (Fintype.card V - 1) := by
  /-
    V : Type u_1
    inst✝¹ : Fintype V
    inst✝ : DecidableEq V
    ⊢ Top.top.IsRegularOfDegree (HSub.hSub (Fintype.card V) 1)
  -/
  intro v
  /-
    V : Type u_1
    inst✝¹ : Fintype V
    inst✝ : DecidableEq V
    v : V
    ⊢ Eq (Top.top.degree v) (HSub.hSub (Fintype.card V) 1)
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The minimum degree of all vertices (and `0` if there are no vertices).
The key properties of this are given in `exists_minimal_degree_vertex`, `minDegree_le_degree`
and `le_minDegree_of_forall_le_degree`. -/
def minDegree [DecidableRel G.Adj] : ℕ :=
  WithTop.untop' 0 (univ.image fun v => G.degree v).min


/-- There exists a vertex of minimal degree. Note the assumption of being nonempty is necessary, as
the lemma implies there exists a vertex. -/
theorem exists_minimal_degree_vertex [DecidableRel G.Adj] [Nonempty V] :
    ∃ v, G.minDegree = G.degree v := by
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝² : Fintype V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : Nonempty V
    ⊢ Exists fun v => Eq G.minDegree (G.degree v)
  -/
  obtain ⟨t, ht : _ = _⟩ := min_of_nonempty (univ_nonempty.image fun v => G.degree v)
  /-
    case intro
    V : Type u_1
    G : SimpleGraph V
    inst✝² : Fintype V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : Nonempty V
    t : Nat
    ht : Eq (Finset.image (fun v => G.degree v) Finset.univ).min ↑t
    ⊢ Exists fun v => Eq G.minDegree (G.degree v)
  -/
  obtain ⟨v, _, rfl⟩ := mem_image.mp (mem_of_min ht)
  /-
    case intro.intro.intro
    V : Type u_1
    G : SimpleGraph V
    inst✝² : Fintype V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : Nonempty V
    v : V
    left✝ : Membership.mem Finset.univ v
    ht : Eq (Finset.image (fun v => G.degree v) Finset.univ).min ↑(G.degree v)
    ⊢ Exists fun v => Eq G.minDegree (G.degree v)
  -/
  exact ⟨v, by simp [minDegree, ht]⟩
  /-
    🎉 no goals
  -/


/-- The minimum degree in the graph is at most the degree of any particular vertex. -/
theorem minDegree_le_degree [DecidableRel G.Adj] (v : V) : G.minDegree ≤ G.degree v := by
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    v : V
    ⊢ LE.le G.minDegree (G.degree v)
  -/
  obtain ⟨t, ht⟩ := Finset.min_of_mem (mem_image_of_mem (fun v => G.degree v) (mem_univ v))
  /-
    case intro
    V : Type u_1
    G : SimpleGraph V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    v : V
    t : Nat
    ht : Eq (Finset.image (fun v => G.degree v) Finset.univ).min ↑t
    ⊢ LE.le G.minDegree (G.degree v)
  -/
  have := Finset.min_le_of_eq (mem_image_of_mem _ (mem_univ v)) ht
  /-
    case intro
    V : Type u_1
    G : SimpleGraph V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    v : V
    t : Nat
    ht : Eq (Finset.image (fun v => G.degree v) Finset.univ).min ↑t
    this : LE.le t (G.degree v)
    ⊢ LE.le G.minDegree (G.degree v)
  -/
  rwa [minDegree, ht]
  /-
    🎉 no goals
  -/


/-- In a nonempty graph, if `k` is at most the degree of every vertex, it is at most the minimum
degree. Note the assumption that the graph is nonempty is necessary as long as `G.minDegree` is
defined to be a natural. -/
theorem le_minDegree_of_forall_le_degree [DecidableRel G.Adj] [Nonempty V] (k : ℕ)
    (h : ∀ v, k ≤ G.degree v) : k ≤ G.minDegree := by
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝² : Fintype V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : Nonempty V
    k : Nat
    h : ∀ (v : V), LE.le k (G.degree v)
    ⊢ LE.le k G.minDegree
  -/
  rcases G.exists_minimal_degree_vertex with ⟨v, hv⟩
  /-
    case intro
    V : Type u_1
    G : SimpleGraph V
    inst✝² : Fintype V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : Nonempty V
    k : Nat
    h : ∀ (v : V), LE.le k (G.degree v)
    v : V
    hv : Eq G.minDegree (G.degree v)
    ⊢ LE.le k G.minDegree
  -/
  rw [hv]
  /-
    case intro
    V : Type u_1
    G : SimpleGraph V
    inst✝² : Fintype V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : Nonempty V
    k : Nat
    h : ∀ (v : V), LE.le k (G.degree v)
    v : V
    hv : Eq G.minDegree (G.degree v)
    ⊢ LE.le k (G.degree v)
  -/
  apply h
  /-
    🎉 no goals
  -/


/-- The maximum degree of all vertices (and `0` if there are no vertices).
The key properties of this are given in `exists_maximal_degree_vertex`, `degree_le_maxDegree`
and `maxDegree_le_of_forall_degree_le`. -/
def maxDegree [DecidableRel G.Adj] : ℕ :=
  Option.getD (univ.image fun v => G.degree v).max 0


/-- There exists a vertex of maximal degree. Note the assumption of being nonempty is necessary, as
the lemma implies there exists a vertex. -/
theorem exists_maximal_degree_vertex [DecidableRel G.Adj] [Nonempty V] :
    ∃ v, G.maxDegree = G.degree v := by
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝² : Fintype V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : Nonempty V
    ⊢ Exists fun v => Eq G.maxDegree (G.degree v)
  -/
  obtain ⟨t, ht⟩ := max_of_nonempty (univ_nonempty.image fun v => G.degree v)
  /-
    case intro
    V : Type u_1
    G : SimpleGraph V
    inst✝² : Fintype V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : Nonempty V
    t : Nat
    ht : Eq (Finset.image (fun v => G.degree v) Finset.univ).max ↑t
    ⊢ Exists fun v => Eq G.maxDegree (G.degree v)
  -/
  have ht₂ := mem_of_max ht
  /-
    case intro
    V : Type u_1
    G : SimpleGraph V
    inst✝² : Fintype V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : Nonempty V
    t : Nat
    ht : Eq (Finset.image (fun v => G.degree v) Finset.univ).max ↑t
    ht₂ : Membership.mem (Finset.image (fun v => G.degree v) Finset.univ) t
    ⊢ Exists fun v => Eq G.maxDegree (G.degree v)
  -/
  simp only [mem_image, mem_univ, exists_prop_of_true] at ht₂
  /-
    case intro
    V : Type u_1
    G : SimpleGraph V
    inst✝² : Fintype V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : Nonempty V
    t : Nat
    ht : Eq (Finset.image (fun v => G.degree v) Finset.univ).max ↑t
    ht₂ : Exists fun a => And True (Eq (G.degree a) t)
    ⊢ Exists fun v => Eq G.maxDegree (G.degree v)
  -/
  rcases ht₂ with ⟨v, _, rfl⟩
  /-
    case intro.intro.intro
    V : Type u_1
    G : SimpleGraph V
    inst✝² : Fintype V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : Nonempty V
    v : V
    left✝ : True
    ht : Eq (Finset.image (fun v => G.degree v) Finset.univ).max ↑(G.degree v)
    ⊢ Exists fun v => Eq G.maxDegree (G.degree v)
  -/
  refine ⟨v, ?_⟩
  /-
    case intro.intro.intro
    V : Type u_1
    G : SimpleGraph V
    inst✝² : Fintype V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : Nonempty V
    v : V
    left✝ : True
    ht : Eq (Finset.image (fun v => G.degree v) Finset.univ).max ↑(G.degree v)
    ⊢ Eq G.maxDegree (G.degree v)
  -/
  rw [maxDegree, ht]
  /-
    case intro.intro.intro
    V : Type u_1
    G : SimpleGraph V
    inst✝² : Fintype V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : Nonempty V
    v : V
    left✝ : True
    ht : Eq (Finset.image (fun v => G.degree v) Finset.univ).max ↑(G.degree v)
    ⊢ Eq (Option.getD (↑(G.degree v)) 0) (G.degree v)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The maximum degree in the graph is at least the degree of any particular vertex. -/
theorem degree_le_maxDegree [DecidableRel G.Adj] (v : V) : G.degree v ≤ G.maxDegree := by
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    v : V
    ⊢ LE.le (G.degree v) G.maxDegree
  -/
  obtain ⟨t, ht : _ = _⟩ := Finset.max_of_mem (mem_image_of_mem (fun v => G.degree v) (mem_univ v))
  /-
    case intro
    V : Type u_1
    G : SimpleGraph V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    v : V
    t : Nat
    ht : Eq (Finset.image (fun v => G.degree v) Finset.univ).max ↑t
    ⊢ LE.le (G.degree v) G.maxDegree
  -/
  have := Finset.le_max_of_eq (mem_image_of_mem _ (mem_univ v)) ht
  /-
    case intro
    V : Type u_1
    G : SimpleGraph V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    v : V
    t : Nat
    ht : Eq (Finset.image (fun v => G.degree v) Finset.univ).max ↑t
    this : LE.le (G.degree v) t
    ⊢ LE.le (G.degree v) G.maxDegree
  -/
  rwa [maxDegree, ht]
  /-
    🎉 no goals
  -/


/-- In a graph, if `k` is at least the degree of every vertex, then it is at least the maximum
degree. -/
theorem maxDegree_le_of_forall_degree_le [DecidableRel G.Adj] (k : ℕ) (h : ∀ v, G.degree v ≤ k) :
    G.maxDegree ≤ k := by
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    k : Nat
    h : ∀ (v : V), LE.le (G.degree v) k
    ⊢ LE.le G.maxDegree k
  -/
  by_cases hV : (univ : Finset V).Nonempty
    /-
      case pos
      V : Type u_1
      G : SimpleGraph V
      inst✝¹ : Fintype V
      inst✝ : DecidableRel G.Adj
      k : Nat
      h : ∀ (v : V), LE.le (G.degree v) k
      hV : Finset.univ.Nonempty
      ⊢ LE.le G.maxDegree k
    -/
  · haveI : Nonempty V := univ_nonempty_iff.mp hV
    /-
      case pos
      V : Type u_1
      G : SimpleGraph V
      inst✝¹ : Fintype V
      inst✝ : DecidableRel G.Adj
      k : Nat
      h : ∀ (v : V), LE.le (G.degree v) k
      hV : Finset.univ.Nonempty
      this : Nonempty V
      ⊢ LE.le G.maxDegree k
    -/
    obtain ⟨v, hv⟩ := G.exists_maximal_degree_vertex
    /-
      case pos.intro
      V : Type u_1
      G : SimpleGraph V
      inst✝¹ : Fintype V
      inst✝ : DecidableRel G.Adj
      k : Nat
      h : ∀ (v : V), LE.le (G.degree v) k
      hV : Finset.univ.Nonempty
      this : Nonempty V
      v : V
      hv : Eq G.maxDegree (G.degree v)
      ⊢ LE.le G.maxDegree k
    -/
    rw [hv]
    /-
      case pos.intro
      V : Type u_1
      G : SimpleGraph V
      inst✝¹ : Fintype V
      inst✝ : DecidableRel G.Adj
      k : Nat
      h : ∀ (v : V), LE.le (G.degree v) k
      hV : Finset.univ.Nonempty
      this : Nonempty V
      v : V
      hv : Eq G.maxDegree (G.degree v)
      ⊢ LE.le (G.degree v) k
    -/
    apply h
    /-
      🎉 no goals
    -/
    /-
      case neg
      V : Type u_1
      G : SimpleGraph V
      inst✝¹ : Fintype V
      inst✝ : DecidableRel G.Adj
      k : Nat
      h : ∀ (v : V), LE.le (G.degree v) k
      hV : Not Finset.univ.Nonempty
      ⊢ LE.le G.maxDegree k
    -/
  · rw [not_nonempty_iff_eq_empty] at hV
    /-
      case neg
      V : Type u_1
      G : SimpleGraph V
      inst✝¹ : Fintype V
      inst✝ : DecidableRel G.Adj
      k : Nat
      h : ∀ (v : V), LE.le (G.degree v) k
      hV : Eq Finset.univ EmptyCollection.emptyCollection
      ⊢ LE.le G.maxDegree k
    -/
    rw [maxDegree, hV, image_empty]
    /-
      case neg
      V : Type u_1
      G : SimpleGraph V
      inst✝¹ : Fintype V
      inst✝ : DecidableRel G.Adj
      k : Nat
      h : ∀ (v : V), LE.le (G.degree v) k
      hV : Eq Finset.univ EmptyCollection.emptyCollection
      ⊢ LE.le (Option.getD EmptyCollection.emptyCollection.max 0) k
    -/
    exact k.zero_le
    /-
      🎉 no goals
    -/


theorem degree_lt_card_verts [DecidableRel G.Adj] (v : V) : G.degree v < Fintype.card V := by
  classical
  apply Finset.card_lt_card
  rw [Finset.ssubset_iff]
  exact ⟨v, by simp, Finset.subset_univ _⟩


/--
The maximum degree of a nonempty graph is less than the number of vertices. Note that the assumption
that `V` is nonempty is necessary, as otherwise this would assert the existence of a
natural number less than zero. -/
theorem maxDegree_lt_card_verts [DecidableRel G.Adj] [Nonempty V] :
    G.maxDegree < Fintype.card V := by
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝² : Fintype V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : Nonempty V
    ⊢ LT.lt G.maxDegree (Fintype.card V)
  -/
  cases' G.exists_maximal_degree_vertex with v hv
  /-
    case intro
    V : Type u_1
    G : SimpleGraph V
    inst✝² : Fintype V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : Nonempty V
    v : V
    hv : Eq G.maxDegree (G.degree v)
    ⊢ LT.lt G.maxDegree (Fintype.card V)
  -/
  rw [hv]
  /-
    case intro
    V : Type u_1
    G : SimpleGraph V
    inst✝² : Fintype V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : Nonempty V
    v : V
    hv : Eq G.maxDegree (G.degree v)
    ⊢ LT.lt (G.degree v) (Fintype.card V)
  -/
  apply G.degree_lt_card_verts v
  /-
    🎉 no goals
  -/


theorem card_commonNeighbors_le_degree_left [DecidableRel G.Adj] (v w : V) :
    Fintype.card (G.commonNeighbors v w) ≤ G.degree v := by
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    v w : V
    ⊢ LE.le (Fintype.card ↑(G.commonNeighbors v w)) (G.degree v)
  -/
  rw [← card_neighborSet_eq_degree]
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    v w : V
    ⊢ LE.le (Fintype.card ↑(G.commonNeighbors v w)) (Fintype.card ↑(G.neighborSet  …
  -/
  exact Set.card_le_card Set.inter_subset_left
  /-
    🎉 no goals
  -/


theorem card_commonNeighbors_le_degree_right [DecidableRel G.Adj] (v w : V) :
    Fintype.card (G.commonNeighbors v w) ≤ G.degree w := by
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    v w : V
    ⊢ LE.le (Fintype.card ↑(G.commonNeighbors v w)) (G.degree w)
  -/
  simp_rw [commonNeighbors_symm _ v w, card_commonNeighbors_le_degree_left]
  /-
    🎉 no goals
  -/


theorem card_commonNeighbors_lt_card_verts [DecidableRel G.Adj] (v w : V) :
    Fintype.card (G.commonNeighbors v w) < Fintype.card V :=
  Nat.lt_of_le_of_lt (G.card_commonNeighbors_le_degree_left _ _) (G.degree_lt_card_verts v)


/-- If the condition `G.Adj v w` fails, then `card_commonNeighbors_le_degree` is
the best we can do in general. -/
theorem Adj.card_commonNeighbors_lt_degree {G : SimpleGraph V} [DecidableRel G.Adj] {v w : V}
    (h : G.Adj v w) : Fintype.card (G.commonNeighbors v w) < G.degree v := by
  classical
  rw [← Set.toFinset_card]
  apply Finset.card_lt_card
  rw [Finset.ssubset_iff]
  use w
  constructor
  · rw [Set.mem_toFinset]
    apply not_mem_commonNeighbors_right
  · rw [Finset.insert_subset_iff]
    constructor
    · simpa
    · rw [neighborFinset, Set.toFinset_subset_toFinset]
      exact G.commonNeighbors_subset_neighborSet_left _ _


theorem card_commonNeighbors_top [DecidableEq V] {v w : V} (h : v ≠ w) :
    Fintype.card ((⊤ : SimpleGraph V).commonNeighbors v w) = Fintype.card V - 2 := by
  /-
    V : Type u_1
    inst✝¹ : Fintype V
    inst✝ : DecidableEq V
    v w : V
    h : Ne v w
    ⊢ Eq (Fintype.card ↑(Top.top.commonNeighbors v w)) (HSub.hSub (Fintype.card V) …
  -/
  simp only [commonNeighbors_top_eq, ← Set.toFinset_card, Set.toFinset_diff]
  /-
    V : Type u_1
    inst✝¹ : Fintype V
    inst✝ : DecidableEq V
    v w : V
    h : Ne v w
    ⊢ Eq (SDiff.sdiff Set.univ.toFinset (Insert.insert v (Singleton.singleton w)). …
  -/
  rw [Finset.card_sdiff]
    /-
      V : Type u_1
      inst✝¹ : Fintype V
      inst✝ : DecidableEq V
      v w : V
      h : Ne v w
      ⊢ Eq (HSub.hSub Set.univ.toFinset.card (Insert.insert v (Singleton.singleton w …
    -/
  · simp [Finset.card_univ, h]
    /-
      🎉 no goals
    -/
    /-
      V : Type u_1
      inst✝¹ : Fintype V
      inst✝ : DecidableEq V
      v w : V
      h : Ne v w
      ⊢ HasSubset.Subset (Insert.insert v (Singleton.singleton w)).toFinset Set.univ …
    -/
  · simp only [Set.toFinset_subset_toFinset, Set.subset_univ]
    /-
      🎉 no goals
    -/


